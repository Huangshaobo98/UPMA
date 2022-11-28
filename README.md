# 无人机联合众包节点作aoi优化
## 目前进度
1. 网络模型（完成）
2. dqn模型适配（完成）
3. 能量模型适配（完成）
4. bug fix（部分完成） 
5. aoi模型(部分完成)
6. 数据持久化工作(完成)

## 待跟进
1. 模型验证
2. 信任模型(可能要使用以前论文的实现)
3. worker的运动模型
4. agent模型目前仍旧无法正确得到训练，仍旧在定位bug中

## 启动
```shell
# 默认启动时进行重新训练
./main.py 
# 可以以追加指令的方式训练
./main.py --train # (or --train-true) 
# 断点续训
./main.py --train --continue_train # (or --continue_train-true)
# 开启测试模式
./main.py --test # (or --train-false)
# 训练/测试数据分析
./main.py --analysis --train # (or -a)
# 开启控制台日志
./main.py --console_log # (or -c, -ct, --console_log-true) 
# 开启文件日志
./main.py --file_log # (or -f,-ft --file_log-true) 