
tar -zxvf /hy-public/Argoverse/forecasting_train_v1.1.tar.gz -C /hy-tmp/argoverse1

cd /root/LaneGCN-unofficial

python data.py --start_iter 0 --workers 8 --batch_size 64 --data_type train

if [ $? -eq 0 ];then
    rm -r /hy-tmp/argoverse1/train
    zip -r /hy-tmp/argoverse1/preprocess/train.zip /hy-tmp/argoverse1/preprocess/train
    gpushare-cli baidu up /hy-tmp/argoverse1/preprocess/train.zip  /孔昊私人/数据集/argoverse1/
    gpushare-cli ali up /hy-tmp/argoverse1/preprocess/train.zip  /轨迹预测数据集/argoverse1/
    oss cp /hy-tmp/argoverse1/preprocess/train.zip oss:///datasets
fi

python data.py --start_iter 0 --workers 8 --batch_size 64 --data_type val

if [ $? -eq 0 ];then
    rm -r /hy-tmp/argoverse1/val
    zip -r /hy-tmp/argoverse1/preprocess/train.zip /hy-tmp/argoverse1/preprocess/val
    gpushare-cli baidu up /hy-tmp/argoverse1/preprocess/train.zip  /孔昊私人/数据集/argoverse1/
    gpushare-cli ali up /hy-tmp/argoverse1/preprocess/train.zip  /轨迹预测数据集/argoverse1/
fi

shutdown