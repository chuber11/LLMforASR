
mkdir -p data

for lang in EN DE
do

    for split in train dev
    do
        cat /project/asr_systems/LT2022/data/$lang/cv14.0/download/${split}_length.mp3.stm | cut -d$'\t' -f1,2 | sed "s/\t/ /g" > data/cv.$lang.$split.seg.aligned
        cat /project/asr_systems/LT2022/data/$lang/cv14.0/download/${split}_length.mp3.stm | cut -d$'\t' -f6 > data/cv.$lang.$split.cased
    done

done

mkdir -p data_test

for lang in EN DE
do

    for split in test
    do
        cat /project/asr_systems/LT2022/data/$lang/cv14.0/download/${split}_length.mp3.stm | cut -d$'\t' -f1,2 | sed "s/\t/ /g" > data_test/cv.$lang.$split.seg.aligned
        cat /project/asr_systems/LT2022/data/$lang/cv14.0/download/${split}_length.mp3.stm | cut -d$'\t' -f6 > data_test/cv.$lang.$split.cased
    done

done
