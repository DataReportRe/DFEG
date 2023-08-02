for((i=0;i<100;i++))
do
	./defped.py --defped_sampling functions_to_test.txt --rid $i
done

cd ../xscope

for((i=1;i<3;i++))
do
	./xscope.py functions_to_test.txt -rid $i
done
