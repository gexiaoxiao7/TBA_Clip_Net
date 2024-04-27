python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part2.txt
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part3.txt
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part4.txt
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part5.txt
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part6.txt
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part7.txt
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part8.txt
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part9.txt
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part10.txt
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part11.txt
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part12.txt


python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --zs 1

python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --zs 1 --only_label 1

python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --zs 1 --if_teacher 0

python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --arch ViT-L/14
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --arch ViT-B/16
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --arch ViT-B/32


python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --load_attention 0 --load_lp 0 --shots 1
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --load_attention 0 --load_lp 0 --shots 2
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --load_attention 0 --load_lp 0 --shots 4

python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --temporal_pooling dd --load_lp 0 --shots 1
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --temporal_pooling dd --load_lp 0 --shots 2
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --temporal_pooling dd --load_lp 0 --shots 4
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --temporal_pooling dd

python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --load_attention 0 --lp 0 --shots 1
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --load_attention 0 --lp 0 --shots 2
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --load_attention 0 --lp 0 --shots 4
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --lp 0

python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --load_attention 0 --load_lp 0 --shots 1 --label_smooth 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --load_attention 0 --load_lp 0 --shots 2 --label_smooth 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --load_attention 0 --load_lp 0 --shots 4 --label_smooth 0
python main.py -cfg ./configs/few_shot/TBAD-8/tba_clip_tbad_few_shot.yaml --test_file datasets_splits/TBAD-8/test_reordered_part1.txt --load_attention 0 --load_lp 0 --label_smooth 0


shutdown -h now