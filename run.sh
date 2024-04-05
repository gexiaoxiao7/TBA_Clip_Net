# 默认最佳设置
# num_frames = 16, arc =  Vit-L/14  prefix, cache_size = 8, shots = 8 , temproal_pooling = attention
# zs = 0

# base
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --load_attention 0 --load_cache 0 --load_pre_feat 0

# zero-shot
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --zs 1

# 不同num_frames
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --num_frames 8
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --num_frames 32

# 不同backbone
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --arch ViT-B/16
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --arch ViT-B/32
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --arch ViT-L/14@336px

#不同cache_size
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --cache_size 1 --load_cache 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --cache_size 2 --load_cache 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --cache_size 3 --load_cache 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --cache_size 4 --load_cache 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --cache_size 5 --load_cache 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --cache_size 6 --load_cache 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --cache_size 7 --load_cache 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --cache_size 9 --load_cache 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --cache_size 10 --load_cache 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --cache_size 11 --load_cache 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --cache_size 12 --load_cache 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --cache_size 13 --load_cache 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --cache_size 14 --load_cache 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --cache_size 15 --load_cache 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --cache_size 16 --load_cache 0

# 不同shots
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --shots 1 --load_attention 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --shots 2 --load_attention 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --shots 3 --load_attention 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --shots 4 --load_attention 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --shots 5 --load_attention 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --shots 6 --load_attention 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --shots 7 --load_attention 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --shots 9 --load_attention 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --shots 10 --load_attention 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --shots 11 --load_attention 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --shots 12 --load_attention 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --shots 13 --load_attention 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --shots 14 --load_attention 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --shots 15 --load_attention 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --shots 16 --load_attention 0

# 是否tld
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --if_teacher 0

# 采用mean_pooling
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --temporal_pooling mean --zs 1
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --temporal_pooling mean

# 采用不同的prefix
# 手工修改


