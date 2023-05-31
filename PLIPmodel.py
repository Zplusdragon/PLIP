from visual_model import *
from SwinViT import swin_base_patch4_window7_224
from textual_model import Textual_encoder

class PLIP_ViT(nn.Module):
    def __init__(self, image_encoder,text_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def forward(self, image,text,masks):
        global_image_out,part_image_out = self.image_encoder(image)
        global_text_out, part_text_out = self.text_encoder(text,masks)
        return global_image_out, part_image_out,global_text_out, part_text_out

class PLIP_MResNet(nn.Module):
    def __init__(self, image_encoder,text_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def get_text_global_embedding(self,caption,mask):
        global_text_out = self.text_encoder.get_global_embedding(caption,mask)
        return global_text_out

    def get_image_embeddings(self,image):
        global_image_out, _,_,_,_ = self.image_encoder(image)
        return global_image_out

    def forward(self, image,text,masks):
        global_image_out,x1,x2,x3,x4 = self.image_encoder(image)
        global_text_out, part_text_out = self.text_encoder(text,masks)
        return global_image_out,x1,x2,x3,x4,global_text_out,part_text_out

class PLIP_Swin(nn.Module):
    def __init__(self, image_encoder,text_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def get_text_global_embedding(self,caption,mask):
        global_text_out = self.text_encoder.get_global_embedding(caption,mask)
        return global_text_out

    # def get_text_attnpool_global_embedding(self,caption,mask):
    #     global_text_out = self.text_encoder.get_attnpool_global_embedding(caption,mask)
    #     return global_text_out

    def get_text_local_embedding(self,caption,mask):
        local_text_out = self.text_encoder.get_local_embedding(caption, mask)
        return local_text_out

    def get_image_embeddings(self,image):
        global_image_out, local_image_out= self.image_encoder.get_embeddings(image)
        return global_image_out,local_image_out

    def get_image_stage_outs(self,image):
        return self.image_encoder.get_stage_outs(image)

    def forward(self, image,text,masks):
        global_image_out,local_image_out,stage_out = self.image_encoder(image)
        global_text_out, part_text_out = self.text_encoder(text,masks)
        return global_image_out,local_image_out,stage_out[0],stage_out[1],stage_out[2],stage_out[3],global_text_out,part_text_out


def Create_PLIP_Model(args):
    if args.plip_model == "MResNet_BERT":
        image_encoder = Image_encoder_ModifiedResNet(args.layers,args.img_dim,args.heads,input_resolution=[args.width,args.height])
        text_encoder = Textual_encoder(encoder_type=args.txt_backbone)
        model = PLIP_MResNet(image_encoder, text_encoder)
        return model
    elif args.plip_model == "MResNetv2_BERT":
        image_encoder = Image_encoder_ModifiedResNetv2(args.layers, args.img_dim, args.heads,
                                                     input_resolution=[args.width, args.height])
        text_encoder = Textual_encoder(encoder_type=args.txt_backbone)
        model = PLIP_MResNet(image_encoder, text_encoder)
        return model
    elif args.plip_model == "ViT_BERT":
        img_backbone = args.img_backbone
        image_encoder = Image_encoder_ViT(backbone=img_backbone)
        text_encoder = Textual_encoder(encoder_type=args.txt_backbone)
        model = PLIP_ViT(image_encoder, text_encoder)
        return model
    elif args.plip_model == "SwinBase_BERT":
        image_encoder = swin_base_patch4_window7_224()
        text_encoder = Textual_encoder(encoder_type=args.txt_backbone)
        model = PLIP_Swin(image_encoder,text_encoder)
        return model
    else:
        raise RuntimeError(f"The image backbone you input does not meet the specification!")