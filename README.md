### Set up 
```bash
conda env create -f environment.yml
```

### Parameter: 
Edit parameters in ***params.py*** file 


### Running 
```bash
python craft_api.py
```
### API 
http://_host_:_port_/query_box?url=<img_path> \
Example: http://localhost:1915/query_box?url=test_imgs/blx_1.jpg\
 - If horizontal_mode is true:
   - if ratio_box_horizontal > 30% --> only horizontal box 
+ box format: [ [left_1, top_1], [right_1, top_2], [right_2, bottom_1], [left_2, bottom_2] ]

