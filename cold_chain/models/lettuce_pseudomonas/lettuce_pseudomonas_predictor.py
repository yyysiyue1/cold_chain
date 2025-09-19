# -*- coding: utf-8 -*-
"""
lettuce_pseudomonas_predictor.py
生菜假单胞菌预测器
用于集成到冷链监控系统中
"""

import pickle
import pandas as pd
import numpy as np
import os


class LettucePseudomonasPredictor:
    """生菜假单胞菌预测器类"""

    def __init__(self, model_path="lettuce_pseudomonas_model.pkl"):
        """
        初始化预测器，加载模型

        参数:
            model_path: pkl模型文件路径
        """
        # 初始化所有属性
        self.model_components = None
        self.lgbm_model = None
        self.scaler = None
        self.poly_transformer = None
        self.gompertz_params = None
        self.adaptation_state = 1.0
        self.prev_temp = None

        # 缓存机制
        self.prediction_cache = {}
        self.max_cache_size = 1000

        # 元数据
        self.metadata = {}

        # 加载模型
        self.load_model(model_path)

    def load_model(self, model_path):
        """加载pkl模型文件"""
        try:
            # 构建绝对路径
            abs_path = os.path.join(os.path.dirname(__file__), model_path)

            # 检查文件是否存在
            if not os.path.exists(abs_path):
                # 尝试直接使用提供的路径
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"❌ 模型文件未找到: {model_path}")
                abs_path = model_path

            # 加载模型
            with open(abs_path, 'rb') as f:
                self.model_components = pickle.load(f)

            # 提取各组件
            self.lgbm_model = self.model_components.get('lgbm_model')
            self.scaler = self.model_components.get('scaler')
            self.poly_transformer = self.model_components.get('poly_transformer')
            self.gompertz_params = self.model_components.get('gompertz_params')

            # 初始化状态变量
            self.adaptation_state = self.model_components.get('adaptation_state', 1.0)
            self.prev_temp = self.model_components.get('prev_temp', None)

            # 获取元数据（如果有）
            self.metadata = self.model_components.get('metadata', {})

            print(f"✅ 生菜假单胞菌模型加载成功: {abs_path}")
            if self.metadata:
                print(f"   模型版本: {self.metadata.get('version', 'unknown')}")
                print(f"   训练日期: {self.metadata.get('training_date', 'unknown')}")

            # 验证模型
            if self.validate_model():
                print("   模型验证: 通过")
            else:
                print("   模型验证: 部分组件缺失")

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.lgbm_model = None

    def validate_model(self):
        """
        验证模型组件是否完整

        返回:
            bool: 模型是否有效
        """
        required_components = {
            'lgbm_model': self.lgbm_model,
            'scaler': self.scaler,
            'poly_transformer': self.poly_transformer,
            'gompertz_params': self.gompertz_params
        }

        missing_components = []
        for name, component in required_components.items():
            if component is None:
                missing_components.append(name)

        if missing_components:
            print(f"   缺少组件: {', '.join(missing_components)}")
            return False

        return True

    def interpolate_growth_rate(self, temp):
        """
        根据温度插值获取生长速率

        参数:
            temp: 当前温度（摄氏度）

        返回:
            生长速率 mu
        """
        if not self.gompertz_params:
            return 0.01

        temps = sorted(self.gompertz_params.keys())

        # 温度过低
        if temp <= temps[0]:
            return self.gompertz_params[temps[0]]['mu'] * (temp / temps[0])

        # 温度过高
        elif temp >= temps[-1]:
            return self.gompertz_params[temps[-1]]['mu']

        # 温度在范围内，线性插值
        else:
            for i in range(len(temps) - 1):
                if temps[i] <= temp <= temps[i + 1]:
                    t1, t2 = temps[i], temps[i + 1]
                    mu1 = self.gompertz_params[t1]['mu']
                    mu2 = self.gompertz_params[t2]['mu']
                    # 线性插值
                    mu = mu1 + (mu2 - mu1) * (temp - t1) / (t2 - t1)
                    return mu

        return 0.01  # 默认值

    def predict_single_step(self, prev_time, prev_conc_log10, current_time, current_temp):
        """
        单步预测接口

        参数:
            prev_time: 前一时刻（小时）
            prev_conc_log10: 前一时刻的浓度（log10 CFU/g）
            current_time: 当前时刻（小时）
            current_temp: 当前温度（摄氏度）

        返回:
            current_conc_log10: 当前预测浓度（log10 CFU/g）
        """
        # 检查模型是否加载成功
        if not self.lgbm_model:
            return prev_conc_log10

        # 处理输入数据
        try:
            prev_time = float(prev_time) if prev_time is not None else 0.0
            current_time = float(current_time) if current_time is not None else 0.0
            prev_conc_log10 = float(prev_conc_log10) if prev_conc_log10 is not None else 0.5
            current_temp = float(current_temp) if current_temp is not None else 4.0
        except (ValueError, TypeError) as e:
            print(f"⚠️ 输入数据转换失败: {e}")
            return prev_conc_log10

        # 初始化前一温度
        if self.prev_temp is None:
            self.prev_temp = current_temp

        # 计算时间步长
        time_step = current_time - prev_time

        # 如果时间步长无效，返回原值
        if time_step <= 0:
            return prev_conc_log10

        # ========== 1. Gompertz基础生长速率 ==========
        gompertz_mu = self.interpolate_growth_rate(current_temp)

        # 获取理论最大浓度K
        if self.gompertz_params:
            K_values = [params['K'] for params in self.gompertz_params.values()]
            theoretical_K = np.mean(K_values) if K_values else 10.0
        else:
            theoretical_K = 10.0

        # Gompertz生长速率（考虑接近K时的减速）
        if prev_conc_log10 < theoretical_K:
            relative_position = prev_conc_log10 / theoretical_K
            gompertz_rate = gompertz_mu * (1 - relative_position * 0.5)
        else:
            gompertz_rate = gompertz_mu * 0.1  # 超过K后极慢生长

        # ========== 2. LightGBM预测 ==========
        try:
            input_features = pd.DataFrame({
                "Temp": [current_temp],
                "Time": [current_time],
                "Concentration_log10_prev": [prev_conc_log10]
            })

            # 多项式特征转换
            input_poly = self.poly_transformer.transform(input_features)

            # 标准化
            input_scaled = self.scaler.transform(input_poly)

            # 预测
            lgbm_prediction = self.lgbm_model.predict(input_scaled)[0]

        except Exception as e:
            lgbm_prediction = gompertz_rate  # 失败时使用Gompertz预测

        # ========== 3. 温度变化适应 ==========
        temp_change = current_temp - self.prev_temp

        if abs(temp_change) > 5:  # 剧烈温度变化
            self.adaptation_state *= 0.6
        elif abs(temp_change) > 2:  # 中等温度变化
            self.adaptation_state *= 0.8
        else:  # 温度稳定
            self.adaptation_state = min(1.0, self.adaptation_state * 1.1)

        # 限制适应状态范围
        self.adaptation_state = np.clip(self.adaptation_state, 0.2, 1.0)

        # ========== 4. 混合预测 ==========
        if lgbm_prediction <= 0:
            final_rate = gompertz_rate * self.adaptation_state
        else:
            if abs(temp_change) < 1:
                # 温度稳定，70% LightGBM + 30% Gompertz
                final_rate = 0.7 * lgbm_prediction + 0.3 * gompertz_rate
            else:
                # 温度变化，30% LightGBM + 70% Gompertz
                final_rate = 0.3 * lgbm_prediction + 0.7 * gompertz_rate

            final_rate *= self.adaptation_state

        # ========== 5. 确保最小生长 ==========
        if current_temp > 0:
            min_rate = 0.0005 * (1 + current_temp / 40)
            final_rate = max(final_rate, min_rate)

        # ========== 6. 计算增量并更新 ==========
        increment = final_rate * time_step

        # 防止单步增长过大
        max_increment = 0.3
        increment = min(increment, max_increment)

        # 计算新浓度
        current_conc_log10 = prev_conc_log10 + increment

        # 确保不超过理论最大值太多
        current_conc_log10 = min(current_conc_log10, theoretical_K * 1.1)

        # 更新前一温度
        self.prev_temp = current_temp

        return current_conc_log10

    def predict(self, time_h, temp_c, prev_conc=None):
        """
        简化的预测接口，类似于TVBNPredictor

        参数:
            time_h: 时间（小时）
            temp_c: 温度（摄氏度）
            prev_conc: 前一时刻浓度，如果为None则使用默认值

        返回:
            预测的浓度值（log10 CFU/g）
        """
        if prev_conc is None:
            prev_conc = 0.5  # 默认初始浓度

        # 构建缓存键
        cache_key = (round(time_h, 2), round(temp_c, 1), round(prev_conc, 2))

        # 检查缓存
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        # 调用原有的单步预测
        result = self.predict_single_step(
            prev_time=0,
            prev_conc_log10=prev_conc,
            current_time=time_h,
            current_temp=temp_c
        )

        # 更新缓存（限制大小）
        if len(self.prediction_cache) >= self.max_cache_size:
            # 删除最早的缓存项
            self.prediction_cache.pop(next(iter(self.prediction_cache)))

        self.prediction_cache[cache_key] = result

        return result

    def predict_batch(self, time_points, temperatures, initial_conc=0.5):
        """
        批量预测多个时间点

        参数:
            time_points: 时间点列表（小时）
            temperatures: 对应的温度列表
            initial_conc: 初始浓度

        返回:
            预测浓度列表
        """
        if not self.validate_model():
            return [initial_conc] * len(time_points)

        predictions = []
        current_conc = initial_conc

        self.reset_state()  # 重置状态

        for i, (t, temp) in enumerate(zip(time_points, temperatures)):
            if i == 0:
                predictions.append(current_conc)
            else:
                prev_t = time_points[i - 1]
                current_conc = self.predict_single_step(
                    prev_time=prev_t,
                    prev_conc_log10=current_conc,
                    current_time=t,
                    current_temp=temp
                )
                predictions.append(current_conc)

        return predictions

    def reset_state(self):
        """
        重置内部状态
        用于开始新的预测序列
        """
        self.adaptation_state = 1.0
        self.prev_temp = None
        self.prediction_cache.clear()

    def get_risk_level(self, conc_log10):
        """
        根据浓度判断风险等级

        参数:
            conc_log10: 假单胞菌浓度（log10 CFU/g）

        返回:
            风险等级字符串
        """
        if conc_log10 < 4:
            return '无'
        elif 4 <= conc_log10 < 5:
            return '低'
        elif 5 <= conc_log10 < 6:
            return '中'
        else:
            return '高'

    def clear_cache(self):
        """清空预测缓存"""
        self.prediction_cache.clear()
        print(f"✅ 缓存已清空")


# 测试代码
if __name__ == "__main__":
    """测试预测器功能"""

    print("=" * 60)
    print("生菜假单胞菌预测器测试")
    print("=" * 60)

    # 创建预测器实例
    predictor = LettucePseudomonasPredictor("lettuce_pseudomonas_model.pkl")

    # 测试单步预测
    if predictor.lgbm_model:
        print("\n1. 测试单步预测...")
        print("-" * 40)

        # 测试数据
        test_cases = [
            (0, 0.5, 1, 4),  # 低温
            (1, 0.6, 2, 10),  # 中温
            (2, 0.8, 3, 25),  # 高温
        ]

        for prev_t, prev_c, curr_t, curr_temp in test_cases:
            result = predictor.predict_single_step(
                prev_time=prev_t,
                prev_conc_log10=prev_c,
                current_time=curr_t,
                current_temp=curr_temp
            )
            risk = predictor.get_risk_level(result)

            print(f"时间: {prev_t}→{curr_t}h, 温度: {curr_temp}℃")
            print(f"  浓度: {prev_c:.2f} → {result:.2f} log10 CFU/g")
            print(f"  风险等级: {risk}")
            print()

        # 测试简化接口
        print("\n2. 测试简化预测接口...")
        print("-" * 40)

        for hours in [1, 5, 10, 24]:
            result = predictor.predict(hours, 10, 0.5)
            print(f"时间: {hours}h, 温度: 10℃, 预测: {result:.2f} log10 CFU/g")

        # 测试批量预测
        print("\n3. 测试批量预测...")
        print("-" * 40)

        time_points = [0, 1, 2, 3, 4, 5]
        temperatures = [4, 6, 8, 10, 12, 14]

        results = predictor.predict_batch(time_points, temperatures, initial_conc=0.5)

        for t, temp, conc in zip(time_points, temperatures, results):
            print(f"时间: {t}h, 温度: {temp}℃, 浓度: {conc:.2f}")

    else:
        print("❌ 模型加载失败，无法进行测试")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)