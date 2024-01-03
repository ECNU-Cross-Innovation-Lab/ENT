procdata_dir=$(dirname $0) # revise this to the directory of code preprocssing data
dataset_dir="/Dataset/IEMOCAP" # revise this to the directory of the dataset
train_set=IEMOCAP

. parse_options.sh || exit 1;

python $procdata_dir/preprocess_$train_set.py --prodir $procdata_dir --path $dataset_dir
python mix.py --dataset $train_set --prodir $procdata_dir