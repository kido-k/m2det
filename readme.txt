// dockerfileのビルド
docker build . -t python/m2det

// dockerコンテナで5000番ポートでsrcフォルダをマウントしてコマンド実行
docker run -it -p 5000:5000 -v $(pwd)/src:/src python/m2det /bin/bash

sh make.sh

python3 demo.py -c=configs/m2det512_vgg.py -m=weights/m2det512_vgg.pth