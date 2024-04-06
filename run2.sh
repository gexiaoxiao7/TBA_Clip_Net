python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --if_teacher 0 --zs 1 --temporal_pooling mean

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

#其他数据集
python main.py -cfg ./configs/few_shot/UCF-101/tba_clip_ucf101_few_shot.yaml --load_attention 0 --load_cache 0 --load_pre_feat 0 --test_file 'datasets_splits/UCF-101/test_split1.txt'
python main.py -cfg ./configs/few_shot/UCF-101/tba_clip_ucf101_few_shot.yaml --test_file 'datasets_splits/UCF-101/test_split2.txt'
python main.py -cfg ./configs/few_shot/UCF-101/tba_clip_ucf101_few_shot.yaml --test_file 'datasets_splits/UCF-101/test_split3.txt'
python main.py -cfg ./configs/few_shot/UCF-101/tba_clip_ucf101_few_shot.yaml --test_file 'datasets_splits/UCF-101/test_split4.txt'
python main.py -cfg ./configs/few_shot/UCF-101/tba_clip_ucf101_few_shot.yaml --test_file 'datasets_splits/UCF-101/test_split5.txt'
python main.py -cfg ./configs/few_shot/UCF-101/tba_clip_ucf101_few_shot.yaml --test_file 'datasets_splits/UCF-101/test_split6.txt'
python main.py -cfg ./configs/few_shot/UCF-101/tba_clip_ucf101_few_shot.yaml --test_file 'datasets_splits/UCF-101/test_split7.txt'
python main.py -cfg ./configs/few_shot/UCF-101/tba_clip_ucf101_few_shot.yaml --test_file 'datasets_splits/UCF-101/test_split8.txt'
python main.py -cfg ./configs/few_shot/UCF-101/tba_clip_ucf101_few_shot.yaml --load_attention 0 --load_cache 0 --load_pre_feat 0 --test_file 'datasets_splits/UCF-101/test_split1.txt' --if_teacher 0
python main.py -cfg ./configs/few_shot/UCF-101/tba_clip_ucf101_few_shot.yaml --test_file 'datasets_splits/UCF-101/test_split2.txt' --if_teacher 0
python main.py -cfg ./configs/few_shot/UCF-101/tba_clip_ucf101_few_shot.yaml --test_file 'datasets_splits/UCF-101/test_split3.txt' --if_teacher 0
python main.py -cfg ./configs/few_shot/UCF-101/tba_clip_ucf101_few_shot.yaml --test_file 'datasets_splits/UCF-101/test_split4.txt' --if_teacher 0
python main.py -cfg ./configs/few_shot/UCF-101/tba_clip_ucf101_few_shot.yaml --test_file 'datasets_splits/UCF-101/test_split5.txt' --if_teacher 0
python main.py -cfg ./configs/few_shot/UCF-101/tba_clip_ucf101_few_shot.yaml --test_file 'datasets_splits/UCF-101/test_split6.txt' --if_teacher 0
python main.py -cfg ./configs/few_shot/UCF-101/tba_clip_ucf101_few_shot.yaml --test_file 'datasets_splits/UCF-101/test_split7.txt' --if_teacher 0
python main.py -cfg ./configs/few_shot/UCF-101/tba_clip_ucf101_few_shot.yaml --test_file 'datasets_splits/UCF-101/test_split8.txt' --if_teacher 0
python main.py -cfg ./configs/few_shot/hmdb/tba_clip_hmdb_few_shot.yaml --load_attention 0 --load_cache 0 --load_pre_feat 0 --test_file 'datasets_splits/hmdb51/test_split1.txt'
python main.py -cfg ./configs/few_shot/hmdb/tba_clip_hmdb_few_shot.yaml --test_file 'datasets_splits/hmdb51/test_split2.txt'
python main.py -cfg ./configs/few_shot/hmdb/tba_clip_hmdb_few_shot.yaml --test_file 'datasets_splits/hmdb51/test_split3.txt'
python main.py -cfg ./configs/few_shot/hmdb/tba_clip_hmdb_few_shot.yaml --test_file 'datasets_splits/hmdb51/test_split4.txt'
python main.py -cfg ./configs/few_shot/hmdb/tba_clip_hmdb_few_shot.yaml --load_attention 0 --load_cache 0 --load_pre_feat 0 --test_file 'datasets_splits/hmdb51/test_split1.txt' --if_teacher 0
python main.py -cfg ./configs/few_shot/hmdb/tba_clip_hmdb_few_shot.yaml --test_file 'datasets_splits/hmdb51/test_split2.txt' --if_teacher 0
python main.py -cfg ./configs/few_shot/hmdb/tba_clip_hmdb_few_shot.yaml --test_file 'datasets_splits/hmdb51/test_split3.txt' --if_teacher 0
python main.py -cfg ./configs/few_shot/hmdb/tba_clip_hmdb_few_shot.yaml --test_file 'datasets_splits/hmdb51/test_split4.txt' --if_teacher 0
# 手动
# python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --zs 1 --temporal_pooling mean --prefix