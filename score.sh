
if [ -z "$1" ]; then
    echo "Error: No input path given"
    exit 1  # Exit with a non-zero status to indicate an error
fi

path=`echo $1 | sed "s/\//_/g"`
e_scripts=scripts

mkdir -p hypos/$path

python $e_scripts/clean_lc.py -inf hypos/hypo_$path.txt -i 1 -lc -splitter space -o hypos/$path/hypo_postpr.txt

for lang in EN DE
do
	echo Language: $lang

    if [ -e "data_test/test_length.$lang.cl.stm" ]; then
        stm=/project/asr_systems/LT2022/data/$lang/cv14.0/download/test_length.stm
        python $e_scripts/clean_lc.py -inf $stm -i 5 -lc -splitter tab -o data_test/test_length.$lang.cl.stm
        stm=data_test/test_length.$lang.cl_lc.stm
    fi

	python $e_scripts/wer.py --hypo hypos/$path/hypo_postpr_lc.txt --ref $stm --ref-field 5 --word-stats-file hypos/$path/stats_txt_AE_$lang.txt > hypos/$path/eval_$lang
	tail -n4 hypos/$path/eval_$lang
done

