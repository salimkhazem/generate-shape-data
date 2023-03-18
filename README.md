# Script to generate synthetic shape dataset  
This script allows to generate simple shapes such as (rectangle, square, triangle, circle) and a csv file which embed the different filenames and the corresponding labels 

## Script 
The script takes two input: 
- Number of image per each class (first argument after the script call) 
- Image size (second argument after the script call) 

```bash 
python generate_data.py NUMBER_IMAGES_PER_CLASS IMG_SIZE 
```

## Output 
The generated images will be stored in folder 'dataset' and a csv file 'dataset.csv'
