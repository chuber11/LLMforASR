
clear
cat $1 | grep eval_loss | cut -d" " -f2 | sed "s/,//g" | nl | sort -n -k2,2 | head -n5 && echo "" && cat $1 | grep eval_loss | cut -d" " -f2 | sed "s/,//g" | nl | tail -n1

