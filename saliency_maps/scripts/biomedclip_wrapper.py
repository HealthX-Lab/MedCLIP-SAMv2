"""
The code below wraps the openai clip model to faciliate extracting layers and encoders.
Based on https://github.com/openai/CLIP and 
"""
import copy
import torch
import torch.nn as nn
from functools import partial
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def permute_then_forward(self, x):
#     x = x.permute(1, 0, 2)
#     x = x + self.attention(self.ln_1(x))
#     x = x + self.mlp(self.ln_2(x))
#     x = x.permute(1, 0, 2) 
#     return x

def permute_then_forward(self, x):
    x = x.permute(1, 0, 2)
    x = x + self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    x = x.permute(1, 0, 2) 
    return x

# class VisionEmbeddings(nn.Module):
#     def __init__(self, class_embedding, patch_embedding, positional_embedding, dtype):
#         super().__init__()
#         self.class_embedding = class_embedding
#         self.patch_embedding = patch_embedding
#         self.positional_embedding = positional_embedding
#         self.dtype = dtype

#     def forward(self, x):
#         x = self.patch_embedding(x.to(self.dtype))  # shape = [*, width, grid, grid]
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         x = torch.cat([self.class_embedding.to(self.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         x = x + self.positional_embedding.to(self.dtype)
#         return x

# class image_encoder_wrapper(nn.Module):
#     def __init__(self, model, dtype):
#         super().__init__()
#         self.transformer = model.trunk
#         self.embeddings = VisionEmbeddings(model.trunk.cls_token,  model.trunk.patch_embed, model.trunk.pos_embed, dtype)
#         self.ln_pre = model.ln_pre
#         self.ln_post = model.ln_post
#         self.proj = model.proj
#         self.dtype = dtype
#         for layer in self.transformer.blocks:
#             layer.forward = partial(permute_then_forward, layer)

#     def forward(self, x, output_hidden_states=False, emb_input = False):
#         if not emb_input:
#             x = self.embeddings(x)
#         x = self.ln_pre(x).to(self.dtype)
#         #x = x.permute(1, 0, 2)  # NLD -> LND
#         hidden_states = [x.clone().detach()]
#         for layer in self.transformer.resblocks:
#             x = layer(x.to(self.dtype))
#             if type(x) == tuple and len(x) == 1: x = x[0]
#             hidden_states.append(x.clone().detach())
#         #x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_post(x[:, 0, :]).type(self.dtype)
#         if self.proj is not None:
#             x = x @ self.proj
#         if output_hidden_states:
#             return {'pooler_output':x, 'hidden_states':hidden_states}
#         else:
#             return x

# class TextEmbeddings(nn.Module):
#     def __init__(self, token_embedding, positional_embedding, dtype):
#         super().__init__()
#         self.token_embedding = token_embedding
#         self.positional_embedding = positional_embedding
#         self.dtype = dtype

#     def forward(self, text):
#         x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
#         x = x + self.positional_embedding.type(self.dtype)[:x.shape[1], :]#(1,50,512)
#         return x


# class text_encoder_wrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.transformer = model.transformer
#         self.embeddings = TextEmbeddings(model.word_embeddings, model.position_embeddings, model.dtype)
#         self.ln_final = model.ln_final
#         self.text_projection = model.text_projection
#         self.dtype = model.dtype
#         for layer in self.transformer.resblocks:
#             layer.attn_mask = None
#             layer.forward = partial(permute_then_forward, layer)
        
    
#     def forward(self, x, output_hidden_states=False, emb_input=False):
#         maxidx = -1 #x.argmax(dim=-1) take features from the eot embedding (eot_token is the highest number in each sequence)
#         if not emb_input:
#             x = self.embeddings(x)
#         #x = x.permute(1, 0, 2)  # NLD -> LND
#         # Insert code to record hidden states
#         hidden_states = [x.clone().detach()] # embedding output
#         for layer in self.transformer.resblocks:
#             x = layer(x.to(self.dtype))
#             if type(x) == tuple and len(x) == 1: x = x[0]
#             hidden_states.append(x.clone().detach())
#         #x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)
#         x = x @ self.text_projection
#         x = x[torch.arange(x.shape[0]), maxidx] @ self.text_projection
#         if output_hidden_states:
#             return {'pooler_output':x, 'hidden_states':hidden_states}
#         else:
#             return x
class VisionEmbeddings(nn.Module):
    def __init__(self, cls_token, patch_embed, pos_embed, dtype):
        super().__init__()
        self.cls_token = cls_token
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed
        self.dtype = dtype

    def forward(self, x):
        x = self.patch_embed(x.to(self.dtype))
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        cls_tokens = cls_tokens.permute(0,2,1)
        x = torch.cat((cls_tokens, x), dim=2).permute(0,2,1)
        x = x + self.pos_embed
        return x

class image_encoder_wrapper(nn.Module):
    def __init__(self, model, dtype):
        super().__init__()
        self.transformer = model.trunk
        self.embeddings = VisionEmbeddings(
            model.trunk.cls_token,
            model.trunk.patch_embed,
            model.trunk.pos_embed,
            dtype
        )
        self.norm = model.trunk.norm
        self.proj = model.head
        self.dtype = dtype
        for layer in self.transformer.blocks:
            layer.forward = partial(permute_then_forward, layer)

    def forward(self, x, output_hidden_states=False, emb_input=False):
        if not emb_input:
            x = self.embeddings(x)
        x = self.norm(x).to(self.dtype)
        hidden_states = [x.clone().detach()]
        for layer in self.transformer.blocks:
            x = layer(x.to(self.dtype))
            hidden_states.append(x.clone().detach())
        x = self.proj(x[:, 0])
        if output_hidden_states:
            return {'pooler_output': x, 'hidden_states': hidden_states}
        else:
            return x

class TextEmbeddings(nn.Module):
    def __init__(self, token_embedding, positional_embedding, dtype):
        super().__init__()
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.dtype = dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)[:x.shape[1], :]
        return x

class text_encoder_wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer = model.text.transformer.encoder
        self.embeddings = TextEmbeddings(
            model.text.transformer.embeddings.word_embeddings,
            model.text.transformer.embeddings.position_embeddings,
            model.text.transformer.dtype
        )
        # self.ln_final = model.text.transformer.encoder.LayerNorm
        self.text_projection = model.text.proj
        self.dtype = model.text.transformer.dtype
        for layer in self.transformer.layer:
            layer.attn_mask = None
            layer.forward = partial(permute_then_forward, layer)

    def forward(self, x, output_hidden_states=False, emb_input=False):
        maxidx = -1
        if not emb_input:
            x = self.embeddings(x)
        hidden_states = [x.clone().detach()]
        for layer in self.transformer.layer:
            x = layer(x.to(self.dtype))
            hidden_states.append(x.clone().detach())
        # x = self.ln_final(x).type(self.dtype)
        x = x @ self.text_projection
        x = x[torch.arange(x.shape[0]), maxidx] @ self.text_projection
        if output_hidden_states:
            return {'pooler_output': x, 'hidden_states': hidden_states}
        else:
            return x

class ClipWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.dtype = model.logit_scale.dtype
        self.vision_model = image_encoder_wrapper(copy.deepcopy(model.visual), self.dtype).to(device)
        print(self.vision_model)
        self.text_model = text_encoder_wrapper(copy.deepcopy(model)).to(device)

    def get_image_features(self, x, output_hidden_states=False, emb_input=False):
        return self.vision_model(x, output_hidden_states, emb_input)

    def get_text_features(self, x, output_hidden_states=False, emb_input=False):
        return self.text_model(x, output_hidden_states, emb_input)

# """
# The code below wraps the openai clip model to faciliate extracting layers and encoders.
# Based on https://github.com/openai/CLIP and 
# """
# import copy
# import torch
# import torch.nn as nn
# from functools import partial
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def permute_then_forward(self, x):
#     x = x.permute(1, 0, 2)
#     x = x + self.attention(self.ln_1(x))
#     x = x + self.mlp(self.ln_2(x))
#     x = x.permute(1, 0, 2) 
#     return x

# class VisionEmbeddings(nn.Module):
#     def __init__(self, class_embedding, patch_embedding, positional_embedding, dtype):
#         super().__init__()
#         self.class_embedding = class_embedding
#         self.patch_embedding = patch_embedding
#         self.positional_embedding = positional_embedding
#         self.dtype = dtype

#     def forward(self, x):
#         x = self.patch_embedding(x.to(self.dtype))  # shape = [*, width, grid, grid]
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         x = torch.cat([self.class_embedding.to(self.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         x = x + self.positional_embedding.to(self.dtype)
#         return x

# class image_encoder_wrapper(nn.Module):
#     def __init__(self, model, dtype):
#         super().__init__()
#         self.transformer = model.trunk
#         self.embeddings = VisionEmbeddings(model.trunk.cls_token,  model.trunk.patch_embed, model.trunk.pos_embed, dtype)
#         self.norm = model.trunk.norm
#         # self.ln_pre = model.ln_pre
#         # self.ln_post = model.ln_post
#         self.proj = model.trunk.proj
#         self.dtype = dtype
#         for layer in self.transformer.blocks:
#             layer.forward = partial(permute_then_forward, layer)

#     def forward(self, x, output_hidden_states=False, emb_input = False):
#         if not emb_input:
#             x = self.embeddings(x)
#         # x = self.ln_pre(x).to(self.dtype)
#         #x = x.permute(1, 0, 2)  # NLD -> LND
#         hidden_states = [x.clone().detach()]
#         for layer in self.transformer.blocks:
#             x = layer(x.to(self.dtype))
#             if type(x) == tuple and len(x) == 1: x = x[0]
#             hidden_states.append(x.clone().detach())
#         #x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.norm(x[:, 0, :]).type(self.dtype)
#         if self.proj is not None:
#             x = x @ self.proj
#         if output_hidden_states:
#             return {'pooler_output':x, 'hidden_states':hidden_states}
#         else:
#             return x

# class TextEmbeddings(nn.Module):
#     def __init__(self, token_embedding, positional_embedding, dtype):
#         super().__init__()
#         self.token_embedding = token_embedding
#         self.positional_embedding = positional_embedding
#         self.dtype = dtype

#     def forward(self, text):
#         x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
#         x = x + self.positional_embedding.type(self.dtype)[:x.shape[1], :]#(1,50,512)
#         return x

# class text_encoder_wrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.transformer = model.transformer
#         self.embeddings = TextEmbeddings(model.token_embedding, model.positional_embedding, model.dtype)
#         self.ln_final = model.ln_final
#         self.text_projection = model.text_projection
#         self.dtype = model.dtype
#         for layer in self.transformer.resblocks:
#             layer.attn_mask = None
#             layer.forward = partial(permute_then_forward, layer)
        
    
#     def forward(self, x, output_hidden_states=False, emb_input=False):
#         maxidx = -1 #x.argmax(dim=-1) take features from the eot embedding (eot_token is the highest number in each sequence)
#         if not emb_input:
#             x = self.embeddings(x)
#         #x = x.permute(1, 0, 2)  # NLD -> LND
#         # Insert code to record hidden states
#         hidden_states = [x.clone().detach()] # embedding output
#         for layer in self.transformer.resblocks:
#             x = layer(x.to(self.dtype))
#             if type(x) == tuple and len(x) == 1: x = x[0]
#             hidden_states.append(x.clone().detach())
#         #x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)
#         x = x @ self.text_projection
#         x = x[torch.arange(x.shape[0]), maxidx] @ self.text_projection
#         if output_hidden_states:
#             return {'pooler_output':x, 'hidden_states':hidden_states}
#         else:
#             return x

# class ClipWrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.vision_model = image_encoder_wrapper(copy.deepcopy(model.visual), model.dtype).to(device)
#         self.text_model = text_encoder_wrapper(copy.deepcopy(model)).to(device)
#         self.dtype = model.dtype

#     def get_image_features(self, x, output_hidden_states=False, emb_input=False):
#         return self.vision_model(x, output_hidden_states, emb_input)

#     def get_text_features(self, x, output_hidden_states=False, emb_input=False):
#         return self.text_model(x, output_hidden_states, emb_input)