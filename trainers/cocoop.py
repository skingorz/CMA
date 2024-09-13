import os.path as osp
from collections import OrderedDict
import math
import torch

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from tqdm import tqdm

from utils import MIE, get_visual_prototype

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        
        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)



        # return image_features, label
        
        logits = []

        imag_feature = []
        text_feature = []
        labels = []
        true_sum = 0
        error_sum = 0
        other_sum = 0
        for pts_i, imf_i, lb in zip(prompts, image_features, label):           # 为每个样本构造一组prompts
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = imf_i @ text_features.t()
            true_similarity = logits[lb]
            mask = torch.ones(logits.size(), dtype=torch.bool).cuda()
            mask[lb] = False
            other_class_logits = torch.masked_select(logits, mask)
            max_error = other_class_logits.max()

            true_sum += true_similarity
            error_sum += max_error

            other_sum += logits.mean()


        return true_sum, error_sum, other_sum


            # imag_feature.append(imf_i.unsqueeze(0))
            # text_feature.append(text_features[lb].unsqueeze(0))
            # labels.append(lb.unsqueeze(0))
        #     l_i = logit_scale * imf_i @ text_features.t()
        #     logits.append(l_i)
        # logits = torch.stack(logits)
        
        # if self.prompt_learner.training:
        #     return F.cross_entropy(logits, label), image_features, text_features

        # return torch.cat(imag_feature, dim=0),torch.cat(text_feature, dim=0), torch.cat(labels, dim=0)


@TRAINER_REGISTRY.register()
class CoCoOp(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]
    
    def get_loss(self, image, label):
        loss, _, _ = self.model(image, label)
        loss_summary = {"loss": loss.item()}
        return loss, loss_summary

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                loss, loss_summary = self.get_loss(image, label)
                # loss, _, _ = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss, loss_summary = self.get_loss(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def model_inference(self, input, label=None):
        return self.model(input, label)
    
    @torch.no_grad()
    def collect(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        # print(f"Evaluate on the *{split}* set")

        im_feas = []
        tx_feas = []
        labels = []
        true_sum = 0
        err_sum = 0
        other_sum = 0
        count = 0
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            true, error, other = self.model_inference(input, label)
            true_sum += true
            err_sum += error
            other_sum += other
            count += len(label)

        print(f"true avg: {true_sum/count}", f"err avg: {err_sum/count}", f"other sum: {other_sum/count}")

        #     im_feas.append(image_feature)
        #     tx_feas.append(text_feature)
        #     labels.append(label)
        
        # im_feas = torch.cat(im_feas)
        # tx_feas = torch.cat(tx_feas)
        # labels = torch.cat(labels)
        
        # torch.save({im_feas, tx_feas, labels}, osp.join("features", "cocoop_vne"))

        # results = self.evaluator.evaluate()

        # for k, v in results.items():
        #     tag = f"{split}/{k}"
        #     self.write_scalar(tag, v, self.epoch)

        # return list(results.values())[0]


class CoCoOpMME(CoCoOp):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.MME = MIE()

    def get_entropy_feature(self, image_feature, text_feature, label):
        text_proto = text_feature[label.unique()]
        entropy_feature = torch.cat([image_feature, text_proto])

        return entropy_feature

    def get_loss(self, image, label):
        celoss, image_feature, text_feature = self.model(image, label)
        entropy_feature = self.get_entropy_feature(image_feature, text_feature, label)
        mme = self.MME(entropy_feature)
        loss = (1 - self.cfg.LAMBDA) * celoss + self.cfg.LAMBDA * mme
        loss_summary = {"ce loss": celoss.item(), "mme loss": mme.item()}
        return loss, loss_summary


@TRAINER_REGISTRY.register()
class CoCoOpMME_ALL(CoCoOpMME):
    pass


@TRAINER_REGISTRY.register()
class CoCoOpMME_MODAL(CoCoOpMME):

    def get_entropy_feature(self, image_feature, text_feature, label):
        text_proto = text_feature[label.unique()]
        visual_proto = get_visual_prototype(label, image_feature)
        entropy_feature = torch.cat([visual_proto, text_proto])
        return entropy_feature


@TRAINER_REGISTRY.register()
class CoCoOpMME_CLASSWISE(CoCoOpMME):

    def get_loss(self, image, label):
        if self.cfg.LAMBDA == 0:
            return CoCoOp.get_loss(self, image, label)
        else:
            celoss, visual_feature, text_feature = self.model(image, label)
            # group visual_feature according to label
            unique_labels = torch.unique(label)
            feature_group = []
            for ul in unique_labels:
                mask = label == ul
                visual_group = visual_feature[mask]
                visual_text_group = torch.cat([visual_group, text_feature[ul].unsqueeze(0)])
                feature_group.append(visual_text_group)
            
            # calculate MME for each group and calucalte mean
            mme = []
            for feature in feature_group:
                mme.append(self.MME(feature))
            mme = torch.mean(torch.stack(mme))
            
            # celoss = F.cross_entropy(output, label)

            loss = (1 - self.cfg.LAMBDA) * celoss + self.cfg.LAMBDA * mme

            loss_summary = {"ce loss": celoss.item(), "mme loss": mme.item()}
            
            return loss, loss_summary