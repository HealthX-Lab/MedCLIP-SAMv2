cd biomedclip_finetuning/open_clip/src

## You can change the following parameters like the GPU devices, batch size, training data, epochs, and DHN-NCE loss parameters.

CUDA_VISIBLE_DEVICES=0 python3 -m open_clip_train.main \
    --batch-size 16 \
    --workers 4 \
    --report-to tensorboard \
    --save-frequency 1 \
    --logs="logs" \
    --dataset-type csv \
    --csv-separator="," \
    --train-data data/medpix_dataset/medpix_dataset.csv \
    --csv-img-key filename \
    --csv-caption-key Caption \
    --lr=1e-3 \
    --wd=0.1 \
    --warmup 1000 \
    --epochs=32 \
    --model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --dhnnce-loss \
    --temperature-dhnnce 0.6 \
    --alpha-dhnnce 0.0 \
    --beta1-dhnnce 0.15 \
    --beta2-dhnnce 0.15