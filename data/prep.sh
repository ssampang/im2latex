echo 'Downloading formulas'
wget https://zenodo.org/record/56198/files/im2latex_formulas.lst
echo 'Downloading dataset partitions'
wget https://zenodo.org/record/56198/files/im2latex_train.lst
wget https://zenodo.org/record/56198/files/im2latex_validate.lst
wget https://zenodo.org/record/56198/files/im2latex_test.lst
echo 'Downloading images'
wget https://zenodo.org/record/56198/files/formula_images.tar.gz
tar -xzvf formula_images.tar.gz

echo 'Preprocessing'
cd im2markup
git submodule update --init --recursive
python scripts/preprocessing/preprocess_images.py --input-dir ../formula_images --output-dir ../images_processed

if node -v; then
  printf''
else
  sudo apt-get install nodejs-legacy
fi

python scripts/preprocessing/preprocess_formulas.py --mode normalize --input-file ../im2latex_formulas.lst --output-file ../formulas.norm.lst
python scripts/preprocessing/preprocess_filter.py --filter --image-dir ../images_processed --label-path ../formulas.norm.lst --data-path ../im2latex_train.lst --output-path ../train_filter.lst
python scripts/preprocessing/preprocess_filter.py --filter --image-dir ../images_processed --label-path ../formulas.norm.lst --data-path ../im2latex_validate.lst --output-path ../validate_filter.lst 
python scripts/preprocessing/preprocess_filter.py --no-filter --image-dir ../images_processed --label-path ../formulas.norm.lst --data-path ../im2latex_test.lst --output-path ../test_filter.lst
python scripts/preprocessing/generate_latex_vocab.py --data-path ../train_filter.lst --label-path ../formulas.norm.lst --output-file ../latex_vocab.txt
echo 'Done'
cd ..
