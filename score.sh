
e_scripts=scripts

python $e_scripts/clean_lc.py -inf hypos/hypo.txt -i 1 -lc -splitter space -o hypos/hypo_postpr.txt

for lang in EN DE
do
	echo Language: $lang
	stm=/project/asr_systems/LT2022/data/$lang/cv14.0/download/test_length.stm
	python $e_scripts/clean_lc.py -inf $stm -i 5 -lc -splitter tab -o data_test/test_length.$lang.cl.stm
	stm=data_test/test_length.$lang.cl_lc.stm

	python $e_scripts/wer.py --hypo hypos/hypo_postpr_lc.txt --ref $stm --ref-field 5 --word-stats-file hypos/stats_txt_AE_$lang.txt > hypos/eval_$lang
	tail -n4 hypos/eval_$lang
done

