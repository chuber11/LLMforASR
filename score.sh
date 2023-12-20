
if [ -z "$1" ]; then
    echo "Error: No input path given"
    exit 1  # Exit with a non-zero status to indicate an error
fi

path=`echo $1 | sed "s/\//_/g"`
e_scripts=scripts

python $e_scripts/clean_lc.py -inf hypos/hypo_$path.txt -i 1 -lc -splitter space -o hypos/hypo_${path}_postpr.txt

for lang in EN DE
do
	echo Language: $lang
	stm=/project/asr_systems/LT2022/data/$lang/cv14.0/download/test_length.stm
	python $e_scripts/clean_lc.py -inf $stm -i 5 -lc -splitter tab -o data_test/test_length.$lang.cl.stm
	stm=data_test/test_length.$lang.cl_lc.stm

	python $e_scripts/wer.py --hypo hypos/hypo_${path}_postpr_lc.txt --ref $stm --ref-field 5 --word-stats-file hypos/stats_txt_AE_$lang.$path.txt > hypos/eval_$lang.$path
	tail -n4 hypos/eval_$lang.$path
done

