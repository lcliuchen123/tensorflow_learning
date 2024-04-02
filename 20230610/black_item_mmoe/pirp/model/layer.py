# coding:utf-8

import tensorflow as tf


class LookupLayer(tf.keras.layers.Layer):
    def __init__(self, name, bins):
        super(LookupLayer, self).__init__(name=name)
        # 构造字典
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(range(1, len(bins) + 1), bins), default_value=0)

    def call(self, inputs, **kwargs):
        return self.table.lookup(inputs)


class TowerLayer(tf.keras.layers.Layer):
    def __init__(self, name, units, output_activation=None):
        super(TowerLayer, self).__init__(name=name)

        # layer列表
        self.layers = []
        # 中间层
        for unit in units[:-1]:
            self.layers.append(tf.keras.layers.Dense(unit, activation='relu'))
        # 输出层
        self.layers.append(tf.keras.layers.Dense(units[-1], activation=output_activation))

    def call(self, inputs, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class MMoELayer(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 num_experts,
                 num_tasks,
                 expert_type='dnn',
                 use_expert_bias=True,
                 use_gate_bias=True,
                 expert_activation='relu',
                 gate_activation='softmax',
                 expert_bias_initializer='zeros',
                 gate_bias_initializer='zeros',
                 expert_bias_regularizer=None,
                 gate_bias_regularizer=None,
                 expert_bias_constraint=None,
                 gate_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling',
                 gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None,
                 gate_kernel_regularizer=None,
                 expert_kernel_constraint=None,
                 gate_kernel_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        """
         Method for instantiating MMoE layer.
        :param units: list of number of hidden units
        :param num_experts: Number of experts
        :param num_tasks: Number of tasks
        :param use_expert_bias: Boolean to indicate the usage of bias in the expert weights
        :param use_gate_bias: Boolean to indicate the usage of bias in the gate weights
        :param expert_activation: Activation function of the expert weights
        :param gate_activation: Activation function of the gate weights
        :param expert_bias_initializer: Initializer for the expert bias
        :param gate_bias_initializer: Initializer for the gate bias
        :param expert_bias_regularizer: Regularizer for the expert bias
        :param gate_bias_regularizer: Regularizer for the gate bias
        :param expert_bias_constraint: Constraint for the expert bias
        :param gate_bias_constraint: Constraint for the gate bias
        :param expert_kernel_initializer: Initializer for the expert weights
        :param gate_kernel_initializer: Initializer for the gate weights
        :param expert_kernel_regularizer: Regularizer for the expert weights
        :param gate_kernel_regularizer: Regularizer for the gate weights
        :param expert_kernel_constraint: Constraint for the expert weights
        :param gate_kernel_constraint: Constraint for the gate weights
        :param activity_regularizer: Regularizer for the activity
        :param kwargs: Additional keyword arguments for the Layer class
        """
        super(MMoELayer, self).__init__(**kwargs)

        # Hidden nodes parameter
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Weight parameter
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = tf.keras.initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = tf.keras.initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = tf.keras.initializers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = tf.keras.initializers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = tf.keras.constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = tf.keras.constraints.get(gate_kernel_constraint)

        # Activation parameter
        self.expert_activation = expert_activation
        self.gate_activation = gate_activation

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = tf.keras.initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = tf.keras.initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = tf.keras.regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = tf.keras.regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = tf.keras.constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = tf.keras.constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        self.expert_layers = []
        self.gate_layers = []

        # type of expert layers: dnn
        self.expert_type = expert_type

        for i in range(self.num_tasks):
            self.gate_layers.append(tf.keras.layers.Dense(self.num_experts,
                                                          activation=self.gate_activation,
                                                          use_bias=self.use_gate_bias,
                                                          kernel_initializer=self.gate_kernel_initializer,
                                                          bias_initializer=self.gate_bias_initializer,
                                                          kernel_regularizer=self.gate_kernel_regularizer,
                                                          bias_regularizer=self.gate_bias_regularizer,
                                                          activity_regularizer=None,
                                                          kernel_constraint=self.gate_kernel_constraint,
                                                          bias_constraint=self.gate_bias_constraint))

        for layer_idx in range(len(self.units)):
            for _ in range(self.num_experts):
                self.expert_layers.append(tf.keras.layers.Dense(self.units[layer_idx],
                                                                activation=self.expert_activation,
                                                                use_bias=self.use_expert_bias,
                                                                kernel_initializer=self.expert_kernel_initializer,
                                                                bias_initializer=self.expert_bias_initializer,
                                                                kernel_regularizer=self.expert_kernel_regularizer,
                                                                bias_regularizer=self.expert_bias_regularizer,
                                                                activity_regularizer=None,
                                                                kernel_constraint=self.expert_kernel_constraint,
                                                                bias_constraint=self.expert_bias_constraint))

    def call(self, inputs, **kwargs):
        """
        Method for the forward function of the layer.
        :param inputs: Input tensor
        :param kwargs: Additional keyword arguments for the base method
        :return: A tensor
        """
        expert_outputs, gate_outputs, final_outputs = [], [], []

        # 计算expert的输出
        for expert_idx in range(self.num_experts):
            expert_input = inputs
            for layer_idx in range(len(self.units)):
                expert_layer = self.expert_layers[layer_idx * self.num_experts + expert_idx]
                expert_input = expert_layer(expert_input)
            expert_outputs.append(tf.expand_dims(expert_input, axis=2))
        expert_outputs = tf.concat(expert_outputs, 2)

        for gate_layer in self.gate_layers:
            gate_outputs.append(gate_layer(inputs))

        # Tensor维度：(batch, units, num_experts)
        for gate_output in gate_outputs:
            expanded_gate_output = tf.expand_dims(gate_output, axis=1)
            weighted_expert_output = expert_outputs * tf.keras.backend.repeat_elements(expanded_gate_output,
                                                                                       self.units[-1], axis=1)
            final_outputs.append(tf.reduce_sum(weighted_expert_output, axis=2))

        # 返回列表长度为task数量，每一个Tensor维度：(batch, units[-1])
        return final_outputs

