## Examples of usage:

1. First go to https://drive.google.com/drive/folders/1_iDsEuHQiNXmMbKx6eUW9GFCjc_2pAMh?usp=sharing and download the review_105600.csv(Game of Terraria review dataset)

2. Create a new dataset folder in the current project(as shown below)

![image](https://github.com/Weitingchien/images_repo/blob/master/ex_1.png?raw=true)

### 3. Re-create the same environment based on environment.yml:

conda env create -f environment.yml

### 4. Activate virtual environment:
conda  activate cyberpunk

### 5. Execute:
python terraria_LSTM.py