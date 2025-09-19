# 【项目名称：冷链预测系统】

> 本项目旨在为冷链物流提供数据驱动的解决方案，通过构建预测模型和自动化数据处理，提升冷链管理的效率和可靠性。

## 项目简介

本项目是一个综合性的冷链管理工具，其核心功能包括：
-   **数据处理与预测：** 利用models中`crayfish_tvbn`等多种预测模型以及(`app/features`)中温湿度过期，对冷链数据进行分析和预测。
-   **数据库交互：** 自动执行数据库设置 (`db/database_setup.py`)，确保数据存储的完整性。
-   **警告处理：** 包含一个专门的模块 (`app/warning_processing.py`) 用于处理和响应系统产生的警告。
-   **配置管理：** 所有系统配置都可以通过 `db/config.ini` 文件进行灵活调整。

## 文件结构说明

-   app
    -   `main.py`: 项目主程序入口。
    -   `prediction_logic.py` : 预测处理代码实现，具体对过期温度湿度标志物进行处理。
    -   `warning_processing.py`: 负责处理各类风险警告并将写入数据库。
-   db
-   `database_setup.py`: 数据库初始化和表结构创建脚本。
-   `config.ini`: 项目配置文件，用于数据库连接、API 密钥等设置。

-   models/crayfish_tvbn
    -   `quick_check_tvbn.py` : 测试动态预测模型的文件
    -   `advanced_prediction_models.py`: 集中存放动态预测模型代码。
    -   `crayfish_tvbn_predictor.py`: `tvbn` 预测模型的调用方法。
    -   `tvbn_predictor_model.pkl`: 训练好的 `tvbn` 模型文件。
    -   `train_crayfish_tvbn_model.py`: 用于训练 `tvbn` 模型的脚本。

-   `cold_chain_app.log`: 应用程序运行日志。
-   `README.md`: 本文件，项目说明。
## 安装

要运行本项目，请确保您安装了 Python 3，并安装所有必要的依赖库。

1.  **克隆项目**
    ```bash
    git clone https://github.com/yyysiyue1/cold_chain.git
    cd cold_chain
    ```

2.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

## 使用

1.  **配置**
    在运行项目之前，请根据您的环境修改 `config.ini` 文件，例如配置数据库连接信息。

2.  **运行项目**
    通过以下命令运行主程序：
    ```bash
    python main.py #这是运行所有代码
    ```

3.  **其他脚本**
    -   训练模型：
        ```bash
        python train_tvbn_model.py
        ```
        如若使用新的模型可以在当前文件夹下新建文件并训练好 建立新类就可以外部调用，如tvbn_predictor.py一样
      - 
    -   数据库初始化：
        ```bash
        python database_setup.py
        ```
    -    动态预测方法：
    - 动态预测方法以函数的方式放在了advanced_prediction_models.py  贡献者的动态预测方法可集中写在此处

## 贡献

- TODO：其他贡献者可以将模型加入本系统直接实现半小时动态读取数据并处理后写入数据再预警的一条龙服务
- 如果要添加模型就只用更改两处
- <img width="439" height="579" alt="e9db0966fb05fc7f1087b86b9cc8ecde" src="https://github.com/user-attachments/assets/282e400b-41e6-40cb-a494-7aee45eedd30" />
- 其中 prediction logic.py中execute_prediction_unit只需更改以下部分
  - 1、execute_prediction_unit(row, food_info, engine, predictor_cache) ///目前无需更改
   -  <img width="951" height="644" alt="image" src="https://github.com/user-attachments/assets/56039fc8-e63f-4987-8fb8-c2de5148a378" />


    - 下面这里可以在路由中直接定义好食品分类代码对应的执行函数
    -  <img width="784" height="187" alt="image" src="https://github.com/user-attachments/assets/8de0e45e-b876-4894-a344-5dd7a1c36dc5" />

    -  对模型crayfish_tvbn中的执行函数 execute 起别名 方便区分
    -  <img width="757" height="145" alt="image" src="https://github.com/user-attachments/assets/46d06aca-f547-417f-b17d-6329b4e2d009" />

    -  另外记得在handle_prediction_results中缓存一下对应的模型
    -  <img width="748" height="104" alt="image" src="https://github.com/user-attachments/assets/df5b0d85-07c1-4317-ac41-21f3837c7530" />
           

- 2、find_previous_abnormal_value(order_number, rec_time, tra_code, order_tra_chain, engine, flag="化学") //重要函数
-  这个是找链上前一条数据的含量值 
-   <img width="1160" height="605" alt="image" src="https://github.com/user-attachments/assets/27863b36-ef9b-4815-89c6-d52a016fe46c" />


- 3、find_previous_monitor_time(order_number, rec_time, tra_code, order_tra_chain, engine): ////重要函数
    """
    查找上一条监测数据的时间。
    """
- 这是寻找链上前一条数据的时间 
- <img width="1099" height="730" alt="image" src="https://github.com/user-attachments/assets/924b2967-a316-4e81-a7d2-5ad0a621f510" />

- 第二部分crayfish_tvbn中重要的就是logic.py 里面的全部可以改写换成其他模型
- 执行函数 execute() 这就是上面提到的起别名那个 然后在标志物整体调度函数中路由那里加上就行
- <img width="1295" height="513" alt="image" src="https://github.com/user-attachments/assets/43553759-fbdd-431a-a87c-c851b9725b03" />
<img width="689" height="165" alt="image" src="https://github.com/user-attachments/assets/0dd71420-4df1-4cd1-a940-f725446d8ebe" />


## 许可证 

杨思越  许可证


---









