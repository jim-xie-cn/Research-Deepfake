cd ../
python src/face_resize.py
python src/face_crop.py
python src/feature.py --action=common
python src/feature.py --action=mfs
python src/feature.py --action=lac
python src/analyse.py
