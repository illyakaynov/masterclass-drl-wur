??	
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.1.02unknown8??
?
conv1_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1_14/kernel
{
#conv1_14/kernel/Read/ReadVariableOpReadVariableOpconv1_14/kernel*&
_output_shapes
:*
dtype0
r
conv1_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1_14/bias
k
!conv1_14/bias/Read/ReadVariableOpReadVariableOpconv1_14/bias*
_output_shapes
:*
dtype0
?
conv2_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2_14/kernel
{
#conv2_14/kernel/Read/ReadVariableOpReadVariableOpconv2_14/kernel*&
_output_shapes
: *
dtype0
r
conv2_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2_14/bias
k
!conv2_14/bias/Read/ReadVariableOpReadVariableOpconv2_14/bias*
_output_shapes
: *
dtype0
?
conv3_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv3_14/kernel
{
#conv3_14/kernel/Read/ReadVariableOpReadVariableOpconv3_14/kernel*&
_output_shapes
:  *
dtype0
r
conv3_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3_14/bias
k
!conv3_14/bias/Read/ReadVariableOpReadVariableOpconv3_14/bias*
_output_shapes
: *
dtype0
?
hidden_dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_namehidden_dense_14/kernel
?
*hidden_dense_14/kernel/Read/ReadVariableOpReadVariableOphidden_dense_14/kernel* 
_output_shapes
:
??*
dtype0
?
hidden_dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_namehidden_dense_14/bias
z
(hidden_dense_14/bias/Read/ReadVariableOpReadVariableOphidden_dense_14/bias*
_output_shapes	
:?*
dtype0
?
dense_advantage_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_namedense_advantage_8/kernel
?
,dense_advantage_8/kernel/Read/ReadVariableOpReadVariableOpdense_advantage_8/kernel*
_output_shapes
:	?*
dtype0
?
dense_advantage_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namedense_advantage_8/bias
}
*dense_advantage_8/bias/Read/ReadVariableOpReadVariableOpdense_advantage_8/bias*
_output_shapes
:*
dtype0
?
hidden_dense_value_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*-
shared_namehidden_dense_value_14/kernel
?
0hidden_dense_value_14/kernel/Read/ReadVariableOpReadVariableOphidden_dense_value_14/kernel*
_output_shapes
:	?*
dtype0
?
hidden_dense_value_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namehidden_dense_value_14/bias
?
.hidden_dense_value_14/bias/Read/ReadVariableOpReadVariableOphidden_dense_value_14/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?*
value?*B?* B?*
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
a
	constants
	variables
regularization_losses
trainable_variables
	keras_api
a
	constants
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
h

#kernel
$bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
h

)kernel
*bias
+	variables
,regularization_losses
-trainable_variables
.	keras_api
R
/	variables
0regularization_losses
1trainable_variables
2	keras_api
h

3kernel
4bias
5	variables
6regularization_losses
7trainable_variables
8	keras_api
h

9kernel
:bias
;	variables
<regularization_losses
=trainable_variables
>	keras_api
a
?	constants
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
h

Dkernel
Ebias
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
a
J	constants
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
a
O	constants
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
V
0
1
#2
$3
)4
*5
36
47
98
:9
D10
E11
 
V
0
1
#2
$3
)4
*5
36
47
98
:9
D10
E11
?
	variables
regularization_losses
Tmetrics
Unon_trainable_variables

Vlayers
Wlayer_regularization_losses
trainable_variables
 
 
 
 
 
?
	variables
regularization_losses
Xmetrics
Ynon_trainable_variables
Zlayer_regularization_losses

[layers
trainable_variables
 
 
 
 
?
	variables
regularization_losses
\metrics
]non_trainable_variables
^layer_regularization_losses

_layers
trainable_variables
[Y
VARIABLE_VALUEconv1_14/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1_14/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
 regularization_losses
`metrics
anon_trainable_variables
blayer_regularization_losses

clayers
!trainable_variables
[Y
VARIABLE_VALUEconv2_14/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2_14/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
 

#0
$1
?
%	variables
&regularization_losses
dmetrics
enon_trainable_variables
flayer_regularization_losses

glayers
'trainable_variables
[Y
VARIABLE_VALUEconv3_14/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3_14/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
 

)0
*1
?
+	variables
,regularization_losses
hmetrics
inon_trainable_variables
jlayer_regularization_losses

klayers
-trainable_variables
 
 
 
?
/	variables
0regularization_losses
lmetrics
mnon_trainable_variables
nlayer_regularization_losses

olayers
1trainable_variables
b`
VARIABLE_VALUEhidden_dense_14/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEhidden_dense_14/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
?
5	variables
6regularization_losses
pmetrics
qnon_trainable_variables
rlayer_regularization_losses

slayers
7trainable_variables
db
VARIABLE_VALUEdense_advantage_8/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEdense_advantage_8/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1
 

90
:1
?
;	variables
<regularization_losses
tmetrics
unon_trainable_variables
vlayer_regularization_losses

wlayers
=trainable_variables
 
 
 
 
?
@	variables
Aregularization_losses
xmetrics
ynon_trainable_variables
zlayer_regularization_losses

{layers
Btrainable_variables
hf
VARIABLE_VALUEhidden_dense_value_14/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEhidden_dense_value_14/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

D0
E1
 

D0
E1
?
F	variables
Gregularization_losses
|metrics
}non_trainable_variables
~layer_regularization_losses

layers
Htrainable_variables
 
 
 
 
?
K	variables
Lregularization_losses
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?layers
Mtrainable_variables
 
 
 
 
?
P	variables
Qregularization_losses
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?layers
Rtrainable_variables
 
 
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_state_inputPlaceholder*/
_output_shapes
:?????????TT*
dtype0*$
shape:?????????TT
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_state_inputconv1_14/kernelconv1_14/biasconv2_14/kernelconv2_14/biasconv3_14/kernelconv3_14/biashidden_dense_14/kernelhidden_dense_14/biasdense_advantage_8/kerneldense_advantage_8/biashidden_dense_value_14/kernelhidden_dense_value_14/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*0
f+R)
'__inference_signature_wrapper_220730904
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1_14/kernel/Read/ReadVariableOp!conv1_14/bias/Read/ReadVariableOp#conv2_14/kernel/Read/ReadVariableOp!conv2_14/bias/Read/ReadVariableOp#conv3_14/kernel/Read/ReadVariableOp!conv3_14/bias/Read/ReadVariableOp*hidden_dense_14/kernel/Read/ReadVariableOp(hidden_dense_14/bias/Read/ReadVariableOp,dense_advantage_8/kernel/Read/ReadVariableOp*dense_advantage_8/bias/Read/ReadVariableOp0hidden_dense_value_14/kernel/Read/ReadVariableOp.hidden_dense_value_14/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*+
f&R$
"__inference__traced_save_220731223
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1_14/kernelconv1_14/biasconv2_14/kernelconv2_14/biasconv3_14/kernelconv3_14/biashidden_dense_14/kernelhidden_dense_14/biasdense_advantage_8/kerneldense_advantage_8/biashidden_dense_value_14/kernelhidden_dense_value_14/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*.
f)R'
%__inference__traced_restore_220731271??
?
?
)__inference_conv3_layer_call_fn_220730601

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+??????????????????????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_conv3_layer_call_and_return_conditional_losses_2207305932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
)__inference_conv2_layer_call_fn_220730580

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+??????????????????????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_conv2_layer_call_and_return_conditional_losses_2207305722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
N__inference_dense_advantage_layer_call_and_return_conditional_losses_220730688

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
'__inference_dqn_layer_call_fn_220731027

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_dqn_layer_call_and_return_conditional_losses_2207308262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????TT::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
Q__inference_hidden_dense_value_layer_call_and_return_conditional_losses_220731132

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
a
5__inference_tf_op_layer_add_8_layer_call_fn_220731163
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Y
fTRR
P__inference_tf_op_layer_add_8_layer_call_and_return_conditional_losses_2207307572
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
J
.__inference_flatten_14_layer_call_fn_220731076

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*R
fMRK
I__inference_flatten_14_layer_call_and_return_conditional_losses_2207306472
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
?
Q__inference_hidden_dense_value_layer_call_and_return_conditional_losses_220730724

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
m
Q__inference_tf_op_layer_Mean_8_layer_call_and_return_conditional_losses_220730706

inputs
identityv
Mean_8/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean_8/reduction_indices?
Mean_8Meaninputs!Mean_8/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:?????????*
	keep_dims(2
Mean_8c
IdentityIdentityMean_8:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
?
0__inference_hidden_dense_layer_call_fn_220731094

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*T
fORM
K__inference_hidden_dense_layer_call_and_return_conditional_losses_2207306662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
'__inference_dqn_layer_call_fn_220730841
state_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstate_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_dqn_layer_call_and_return_conditional_losses_2207308262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????TT::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:+ '
%
_user_specified_namestate_input
?8
?
B__inference_dqn_layer_call_and_return_conditional_losses_220730826

inputs(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2(
$conv2_statefulpartitionedcall_args_1(
$conv2_statefulpartitionedcall_args_2(
$conv3_statefulpartitionedcall_args_1(
$conv3_statefulpartitionedcall_args_2/
+hidden_dense_statefulpartitionedcall_args_1/
+hidden_dense_statefulpartitionedcall_args_22
.dense_advantage_statefulpartitionedcall_args_12
.dense_advantage_statefulpartitionedcall_args_25
1hidden_dense_value_statefulpartitionedcall_args_15
1hidden_dense_value_statefulpartitionedcall_args_2
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?'dense_advantage/StatefulPartitionedCall?$hidden_dense/StatefulPartitionedCall?*hidden_dense_value/StatefulPartitionedCall?
#tf_op_layer_Cast_14/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????TT*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*[
fVRT
R__inference_tf_op_layer_Cast_14_layer_call_and_return_conditional_losses_2207306102%
#tf_op_layer_Cast_14/PartitionedCall?
&tf_op_layer_truediv_14/PartitionedCallPartitionedCall,tf_op_layer_Cast_14/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????TT*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*^
fYRW
U__inference_tf_op_layer_truediv_14_layer_call_and_return_conditional_losses_2207306242(
&tf_op_layer_truediv_14/PartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_truediv_14/PartitionedCall:output:0$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_conv1_layer_call_and_return_conditional_losses_2207305512
conv1/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0$conv2_statefulpartitionedcall_args_1$conv2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????		 *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_conv2_layer_call_and_return_conditional_losses_2207305722
conv2/StatefulPartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0$conv3_statefulpartitionedcall_args_1$conv3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_conv3_layer_call_and_return_conditional_losses_2207305932
conv3/StatefulPartitionedCall?
flatten_14/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*R
fMRK
I__inference_flatten_14_layer_call_and_return_conditional_losses_2207306472
flatten_14/PartitionedCall?
$hidden_dense/StatefulPartitionedCallStatefulPartitionedCall#flatten_14/PartitionedCall:output:0+hidden_dense_statefulpartitionedcall_args_1+hidden_dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*T
fORM
K__inference_hidden_dense_layer_call_and_return_conditional_losses_2207306662&
$hidden_dense/StatefulPartitionedCall?
'dense_advantage/StatefulPartitionedCallStatefulPartitionedCall-hidden_dense/StatefulPartitionedCall:output:0.dense_advantage_statefulpartitionedcall_args_1.dense_advantage_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*W
fRRP
N__inference_dense_advantage_layer_call_and_return_conditional_losses_2207306882)
'dense_advantage/StatefulPartitionedCall?
"tf_op_layer_Mean_8/PartitionedCallPartitionedCall0dense_advantage/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Z
fURS
Q__inference_tf_op_layer_Mean_8_layer_call_and_return_conditional_losses_2207307062$
"tf_op_layer_Mean_8/PartitionedCall?
*hidden_dense_value/StatefulPartitionedCallStatefulPartitionedCall-hidden_dense/StatefulPartitionedCall:output:01hidden_dense_value_statefulpartitionedcall_args_11hidden_dense_value_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Z
fURS
Q__inference_hidden_dense_value_layer_call_and_return_conditional_losses_2207307242,
*hidden_dense_value/StatefulPartitionedCall?
!tf_op_layer_Sub_8/PartitionedCallPartitionedCall0dense_advantage/StatefulPartitionedCall:output:0+tf_op_layer_Mean_8/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Y
fTRR
P__inference_tf_op_layer_Sub_8_layer_call_and_return_conditional_losses_2207307422#
!tf_op_layer_Sub_8/PartitionedCall?
!tf_op_layer_add_8/PartitionedCallPartitionedCall3hidden_dense_value/StatefulPartitionedCall:output:0*tf_op_layer_Sub_8/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Y
fTRR
P__inference_tf_op_layer_add_8_layer_call_and_return_conditional_losses_2207307572#
!tf_op_layer_add_8/PartitionedCall?
IdentityIdentity*tf_op_layer_add_8/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall(^dense_advantage/StatefulPartitionedCall%^hidden_dense/StatefulPartitionedCall+^hidden_dense_value/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????TT::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2R
'dense_advantage/StatefulPartitionedCall'dense_advantage/StatefulPartitionedCall2L
$hidden_dense/StatefulPartitionedCall$hidden_dense/StatefulPartitionedCall2X
*hidden_dense_value/StatefulPartitionedCall*hidden_dense_value/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
X
:__inference_tf_op_layer_truediv_14_layer_call_fn_220731065
inputs_0
identity?
PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????TT*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*^
fYRW
U__inference_tf_op_layer_truediv_14_layer_call_and_return_conditional_losses_2207306242
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????TT2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????TT:( $
"
_user_specified_name
inputs/0
?M
?	
$__inference__wrapped_model_220730538
state_input,
(dqn_conv1_conv2d_readvariableop_resource-
)dqn_conv1_biasadd_readvariableop_resource,
(dqn_conv2_conv2d_readvariableop_resource-
)dqn_conv2_biasadd_readvariableop_resource,
(dqn_conv3_conv2d_readvariableop_resource-
)dqn_conv3_biasadd_readvariableop_resource3
/dqn_hidden_dense_matmul_readvariableop_resource4
0dqn_hidden_dense_biasadd_readvariableop_resource6
2dqn_dense_advantage_matmul_readvariableop_resource7
3dqn_dense_advantage_biasadd_readvariableop_resource9
5dqn_hidden_dense_value_matmul_readvariableop_resource:
6dqn_hidden_dense_value_biasadd_readvariableop_resource
identity?? dqn/conv1/BiasAdd/ReadVariableOp?dqn/conv1/Conv2D/ReadVariableOp? dqn/conv2/BiasAdd/ReadVariableOp?dqn/conv2/Conv2D/ReadVariableOp? dqn/conv3/BiasAdd/ReadVariableOp?dqn/conv3/Conv2D/ReadVariableOp?*dqn/dense_advantage/BiasAdd/ReadVariableOp?)dqn/dense_advantage/MatMul/ReadVariableOp?'dqn/hidden_dense/BiasAdd/ReadVariableOp?&dqn/hidden_dense/MatMul/ReadVariableOp?-dqn/hidden_dense_value/BiasAdd/ReadVariableOp?,dqn/hidden_dense_value/MatMul/ReadVariableOp?
dqn/tf_op_layer_Cast_14/Cast_14Caststate_input*

DstT0*

SrcT0*
_cloned(*/
_output_shapes
:?????????TT2!
dqn/tf_op_layer_Cast_14/Cast_14?
'dqn/tf_op_layer_truediv_14/truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2)
'dqn/tf_op_layer_truediv_14/truediv_14/y?
%dqn/tf_op_layer_truediv_14/truediv_14RealDiv#dqn/tf_op_layer_Cast_14/Cast_14:y:00dqn/tf_op_layer_truediv_14/truediv_14/y:output:0*
T0*
_cloned(*/
_output_shapes
:?????????TT2'
%dqn/tf_op_layer_truediv_14/truediv_14?
dqn/conv1/Conv2D/ReadVariableOpReadVariableOp(dqn_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
dqn/conv1/Conv2D/ReadVariableOp?
dqn/conv1/Conv2DConv2D)dqn/tf_op_layer_truediv_14/truediv_14:z:0'dqn/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
dqn/conv1/Conv2D?
 dqn/conv1/BiasAdd/ReadVariableOpReadVariableOp)dqn_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dqn/conv1/BiasAdd/ReadVariableOp?
dqn/conv1/BiasAddBiasAdddqn/conv1/Conv2D:output:0(dqn/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
dqn/conv1/BiasAdd~
dqn/conv1/ReluReludqn/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
dqn/conv1/Relu?
dqn/conv2/Conv2D/ReadVariableOpReadVariableOp(dqn_conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
dqn/conv2/Conv2D/ReadVariableOp?
dqn/conv2/Conv2DConv2Ddqn/conv1/Relu:activations:0'dqn/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 *
paddingVALID*
strides
2
dqn/conv2/Conv2D?
 dqn/conv2/BiasAdd/ReadVariableOpReadVariableOp)dqn_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dqn/conv2/BiasAdd/ReadVariableOp?
dqn/conv2/BiasAddBiasAdddqn/conv2/Conv2D:output:0(dqn/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 2
dqn/conv2/BiasAdd~
dqn/conv2/ReluReludqn/conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????		 2
dqn/conv2/Relu?
dqn/conv3/Conv2D/ReadVariableOpReadVariableOp(dqn_conv3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
dqn/conv3/Conv2D/ReadVariableOp?
dqn/conv3/Conv2DConv2Ddqn/conv2/Relu:activations:0'dqn/conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
dqn/conv3/Conv2D?
 dqn/conv3/BiasAdd/ReadVariableOpReadVariableOp)dqn_conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dqn/conv3/BiasAdd/ReadVariableOp?
dqn/conv3/BiasAddBiasAdddqn/conv3/Conv2D:output:0(dqn/conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
dqn/conv3/BiasAdd~
dqn/conv3/ReluReludqn/conv3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
dqn/conv3/Relu}
dqn/flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
dqn/flatten_14/Const?
dqn/flatten_14/ReshapeReshapedqn/conv3/Relu:activations:0dqn/flatten_14/Const:output:0*
T0*(
_output_shapes
:??????????2
dqn/flatten_14/Reshape?
&dqn/hidden_dense/MatMul/ReadVariableOpReadVariableOp/dqn_hidden_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&dqn/hidden_dense/MatMul/ReadVariableOp?
dqn/hidden_dense/MatMulMatMuldqn/flatten_14/Reshape:output:0.dqn/hidden_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dqn/hidden_dense/MatMul?
'dqn/hidden_dense/BiasAdd/ReadVariableOpReadVariableOp0dqn_hidden_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'dqn/hidden_dense/BiasAdd/ReadVariableOp?
dqn/hidden_dense/BiasAddBiasAdd!dqn/hidden_dense/MatMul:product:0/dqn/hidden_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dqn/hidden_dense/BiasAdd?
dqn/hidden_dense/TanhTanh!dqn/hidden_dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dqn/hidden_dense/Tanh?
)dqn/dense_advantage/MatMul/ReadVariableOpReadVariableOp2dqn_dense_advantage_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02+
)dqn/dense_advantage/MatMul/ReadVariableOp?
dqn/dense_advantage/MatMulMatMuldqn/hidden_dense/Tanh:y:01dqn/dense_advantage/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dqn/dense_advantage/MatMul?
*dqn/dense_advantage/BiasAdd/ReadVariableOpReadVariableOp3dqn_dense_advantage_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*dqn/dense_advantage/BiasAdd/ReadVariableOp?
dqn/dense_advantage/BiasAddBiasAdd$dqn/dense_advantage/MatMul:product:02dqn/dense_advantage/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dqn/dense_advantage/BiasAdd?
/dqn/tf_op_layer_Mean_8/Mean_8/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/dqn/tf_op_layer_Mean_8/Mean_8/reduction_indices?
dqn/tf_op_layer_Mean_8/Mean_8Mean$dqn/dense_advantage/BiasAdd:output:08dqn/tf_op_layer_Mean_8/Mean_8/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:?????????*
	keep_dims(2
dqn/tf_op_layer_Mean_8/Mean_8?
,dqn/hidden_dense_value/MatMul/ReadVariableOpReadVariableOp5dqn_hidden_dense_value_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02.
,dqn/hidden_dense_value/MatMul/ReadVariableOp?
dqn/hidden_dense_value/MatMulMatMuldqn/hidden_dense/Tanh:y:04dqn/hidden_dense_value/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dqn/hidden_dense_value/MatMul?
-dqn/hidden_dense_value/BiasAdd/ReadVariableOpReadVariableOp6dqn_hidden_dense_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-dqn/hidden_dense_value/BiasAdd/ReadVariableOp?
dqn/hidden_dense_value/BiasAddBiasAdd'dqn/hidden_dense_value/MatMul:product:05dqn/hidden_dense_value/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
dqn/hidden_dense_value/BiasAdd?
dqn/tf_op_layer_Sub_8/Sub_8Sub$dqn/dense_advantage/BiasAdd:output:0&dqn/tf_op_layer_Mean_8/Mean_8:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2
dqn/tf_op_layer_Sub_8/Sub_8?
dqn/tf_op_layer_add_8/add_8AddV2'dqn/hidden_dense_value/BiasAdd:output:0dqn/tf_op_layer_Sub_8/Sub_8:z:0*
T0*
_cloned(*'
_output_shapes
:?????????2
dqn/tf_op_layer_add_8/add_8?
IdentityIdentitydqn/tf_op_layer_add_8/add_8:z:0!^dqn/conv1/BiasAdd/ReadVariableOp ^dqn/conv1/Conv2D/ReadVariableOp!^dqn/conv2/BiasAdd/ReadVariableOp ^dqn/conv2/Conv2D/ReadVariableOp!^dqn/conv3/BiasAdd/ReadVariableOp ^dqn/conv3/Conv2D/ReadVariableOp+^dqn/dense_advantage/BiasAdd/ReadVariableOp*^dqn/dense_advantage/MatMul/ReadVariableOp(^dqn/hidden_dense/BiasAdd/ReadVariableOp'^dqn/hidden_dense/MatMul/ReadVariableOp.^dqn/hidden_dense_value/BiasAdd/ReadVariableOp-^dqn/hidden_dense_value/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????TT::::::::::::2D
 dqn/conv1/BiasAdd/ReadVariableOp dqn/conv1/BiasAdd/ReadVariableOp2B
dqn/conv1/Conv2D/ReadVariableOpdqn/conv1/Conv2D/ReadVariableOp2D
 dqn/conv2/BiasAdd/ReadVariableOp dqn/conv2/BiasAdd/ReadVariableOp2B
dqn/conv2/Conv2D/ReadVariableOpdqn/conv2/Conv2D/ReadVariableOp2D
 dqn/conv3/BiasAdd/ReadVariableOp dqn/conv3/BiasAdd/ReadVariableOp2B
dqn/conv3/Conv2D/ReadVariableOpdqn/conv3/Conv2D/ReadVariableOp2X
*dqn/dense_advantage/BiasAdd/ReadVariableOp*dqn/dense_advantage/BiasAdd/ReadVariableOp2V
)dqn/dense_advantage/MatMul/ReadVariableOp)dqn/dense_advantage/MatMul/ReadVariableOp2R
'dqn/hidden_dense/BiasAdd/ReadVariableOp'dqn/hidden_dense/BiasAdd/ReadVariableOp2P
&dqn/hidden_dense/MatMul/ReadVariableOp&dqn/hidden_dense/MatMul/ReadVariableOp2^
-dqn/hidden_dense_value/BiasAdd/ReadVariableOp-dqn/hidden_dense_value/BiasAdd/ReadVariableOp2\
,dqn/hidden_dense_value/MatMul/ReadVariableOp,dqn/hidden_dense_value/MatMul/ReadVariableOp:+ '
%
_user_specified_namestate_input
?8
?
B__inference_dqn_layer_call_and_return_conditional_losses_220730795
state_input(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2(
$conv2_statefulpartitionedcall_args_1(
$conv2_statefulpartitionedcall_args_2(
$conv3_statefulpartitionedcall_args_1(
$conv3_statefulpartitionedcall_args_2/
+hidden_dense_statefulpartitionedcall_args_1/
+hidden_dense_statefulpartitionedcall_args_22
.dense_advantage_statefulpartitionedcall_args_12
.dense_advantage_statefulpartitionedcall_args_25
1hidden_dense_value_statefulpartitionedcall_args_15
1hidden_dense_value_statefulpartitionedcall_args_2
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?'dense_advantage/StatefulPartitionedCall?$hidden_dense/StatefulPartitionedCall?*hidden_dense_value/StatefulPartitionedCall?
#tf_op_layer_Cast_14/PartitionedCallPartitionedCallstate_input*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????TT*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*[
fVRT
R__inference_tf_op_layer_Cast_14_layer_call_and_return_conditional_losses_2207306102%
#tf_op_layer_Cast_14/PartitionedCall?
&tf_op_layer_truediv_14/PartitionedCallPartitionedCall,tf_op_layer_Cast_14/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????TT*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*^
fYRW
U__inference_tf_op_layer_truediv_14_layer_call_and_return_conditional_losses_2207306242(
&tf_op_layer_truediv_14/PartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_truediv_14/PartitionedCall:output:0$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_conv1_layer_call_and_return_conditional_losses_2207305512
conv1/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0$conv2_statefulpartitionedcall_args_1$conv2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????		 *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_conv2_layer_call_and_return_conditional_losses_2207305722
conv2/StatefulPartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0$conv3_statefulpartitionedcall_args_1$conv3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_conv3_layer_call_and_return_conditional_losses_2207305932
conv3/StatefulPartitionedCall?
flatten_14/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*R
fMRK
I__inference_flatten_14_layer_call_and_return_conditional_losses_2207306472
flatten_14/PartitionedCall?
$hidden_dense/StatefulPartitionedCallStatefulPartitionedCall#flatten_14/PartitionedCall:output:0+hidden_dense_statefulpartitionedcall_args_1+hidden_dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*T
fORM
K__inference_hidden_dense_layer_call_and_return_conditional_losses_2207306662&
$hidden_dense/StatefulPartitionedCall?
'dense_advantage/StatefulPartitionedCallStatefulPartitionedCall-hidden_dense/StatefulPartitionedCall:output:0.dense_advantage_statefulpartitionedcall_args_1.dense_advantage_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*W
fRRP
N__inference_dense_advantage_layer_call_and_return_conditional_losses_2207306882)
'dense_advantage/StatefulPartitionedCall?
"tf_op_layer_Mean_8/PartitionedCallPartitionedCall0dense_advantage/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Z
fURS
Q__inference_tf_op_layer_Mean_8_layer_call_and_return_conditional_losses_2207307062$
"tf_op_layer_Mean_8/PartitionedCall?
*hidden_dense_value/StatefulPartitionedCallStatefulPartitionedCall-hidden_dense/StatefulPartitionedCall:output:01hidden_dense_value_statefulpartitionedcall_args_11hidden_dense_value_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Z
fURS
Q__inference_hidden_dense_value_layer_call_and_return_conditional_losses_2207307242,
*hidden_dense_value/StatefulPartitionedCall?
!tf_op_layer_Sub_8/PartitionedCallPartitionedCall0dense_advantage/StatefulPartitionedCall:output:0+tf_op_layer_Mean_8/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Y
fTRR
P__inference_tf_op_layer_Sub_8_layer_call_and_return_conditional_losses_2207307422#
!tf_op_layer_Sub_8/PartitionedCall?
!tf_op_layer_add_8/PartitionedCallPartitionedCall3hidden_dense_value/StatefulPartitionedCall:output:0*tf_op_layer_Sub_8/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Y
fTRR
P__inference_tf_op_layer_add_8_layer_call_and_return_conditional_losses_2207307572#
!tf_op_layer_add_8/PartitionedCall?
IdentityIdentity*tf_op_layer_add_8/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall(^dense_advantage/StatefulPartitionedCall%^hidden_dense/StatefulPartitionedCall+^hidden_dense_value/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????TT::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2R
'dense_advantage/StatefulPartitionedCall'dense_advantage/StatefulPartitionedCall2L
$hidden_dense/StatefulPartitionedCall$hidden_dense/StatefulPartitionedCall2X
*hidden_dense_value/StatefulPartitionedCall*hidden_dense_value/StatefulPartitionedCall:+ '
%
_user_specified_namestate_input
?G
?
B__inference_dqn_layer_call_and_return_conditional_losses_220731010

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource/
+hidden_dense_matmul_readvariableop_resource0
,hidden_dense_biasadd_readvariableop_resource2
.dense_advantage_matmul_readvariableop_resource3
/dense_advantage_biasadd_readvariableop_resource5
1hidden_dense_value_matmul_readvariableop_resource6
2hidden_dense_value_biasadd_readvariableop_resource
identity??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?conv3/BiasAdd/ReadVariableOp?conv3/Conv2D/ReadVariableOp?&dense_advantage/BiasAdd/ReadVariableOp?%dense_advantage/MatMul/ReadVariableOp?#hidden_dense/BiasAdd/ReadVariableOp?"hidden_dense/MatMul/ReadVariableOp?)hidden_dense_value/BiasAdd/ReadVariableOp?(hidden_dense_value/MatMul/ReadVariableOp?
tf_op_layer_Cast_14/Cast_14Castinputs*

DstT0*

SrcT0*
_cloned(*/
_output_shapes
:?????????TT2
tf_op_layer_Cast_14/Cast_14?
#tf_op_layer_truediv_14/truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2%
#tf_op_layer_truediv_14/truediv_14/y?
!tf_op_layer_truediv_14/truediv_14RealDivtf_op_layer_Cast_14/Cast_14:y:0,tf_op_layer_truediv_14/truediv_14/y:output:0*
T0*
_cloned(*/
_output_shapes
:?????????TT2#
!tf_op_layer_truediv_14/truediv_14?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2D%tf_op_layer_truediv_14/truediv_14:z:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1/Conv2D?
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2

conv1/Relu?
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2/Conv2D/ReadVariableOp?
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 *
paddingVALID*
strides
2
conv2/Conv2D?
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2/BiasAdd/ReadVariableOp?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????		 2

conv2/Relu?
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv3/Conv2D/ReadVariableOp?
conv3/Conv2DConv2Dconv2/Relu:activations:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv3/Conv2D?
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3/BiasAdd/ReadVariableOp?
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv3/BiasAddr

conv3/ReluReluconv3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2

conv3/Reluu
flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_14/Const?
flatten_14/ReshapeReshapeconv3/Relu:activations:0flatten_14/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_14/Reshape?
"hidden_dense/MatMul/ReadVariableOpReadVariableOp+hidden_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"hidden_dense/MatMul/ReadVariableOp?
hidden_dense/MatMulMatMulflatten_14/Reshape:output:0*hidden_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_dense/MatMul?
#hidden_dense/BiasAdd/ReadVariableOpReadVariableOp,hidden_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#hidden_dense/BiasAdd/ReadVariableOp?
hidden_dense/BiasAddBiasAddhidden_dense/MatMul:product:0+hidden_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_dense/BiasAdd?
hidden_dense/TanhTanhhidden_dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
hidden_dense/Tanh?
%dense_advantage/MatMul/ReadVariableOpReadVariableOp.dense_advantage_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%dense_advantage/MatMul/ReadVariableOp?
dense_advantage/MatMulMatMulhidden_dense/Tanh:y:0-dense_advantage/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_advantage/MatMul?
&dense_advantage/BiasAdd/ReadVariableOpReadVariableOp/dense_advantage_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&dense_advantage/BiasAdd/ReadVariableOp?
dense_advantage/BiasAddBiasAdd dense_advantage/MatMul:product:0.dense_advantage/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_advantage/BiasAdd?
+tf_op_layer_Mean_8/Mean_8/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf_op_layer_Mean_8/Mean_8/reduction_indices?
tf_op_layer_Mean_8/Mean_8Mean dense_advantage/BiasAdd:output:04tf_op_layer_Mean_8/Mean_8/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:?????????*
	keep_dims(2
tf_op_layer_Mean_8/Mean_8?
(hidden_dense_value/MatMul/ReadVariableOpReadVariableOp1hidden_dense_value_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(hidden_dense_value/MatMul/ReadVariableOp?
hidden_dense_value/MatMulMatMulhidden_dense/Tanh:y:00hidden_dense_value/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
hidden_dense_value/MatMul?
)hidden_dense_value/BiasAdd/ReadVariableOpReadVariableOp2hidden_dense_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)hidden_dense_value/BiasAdd/ReadVariableOp?
hidden_dense_value/BiasAddBiasAdd#hidden_dense_value/MatMul:product:01hidden_dense_value/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
hidden_dense_value/BiasAdd?
tf_op_layer_Sub_8/Sub_8Sub dense_advantage/BiasAdd:output:0"tf_op_layer_Mean_8/Mean_8:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2
tf_op_layer_Sub_8/Sub_8?
tf_op_layer_add_8/add_8AddV2#hidden_dense_value/BiasAdd:output:0tf_op_layer_Sub_8/Sub_8:z:0*
T0*
_cloned(*'
_output_shapes
:?????????2
tf_op_layer_add_8/add_8?
IdentityIdentitytf_op_layer_add_8/add_8:z:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp'^dense_advantage/BiasAdd/ReadVariableOp&^dense_advantage/MatMul/ReadVariableOp$^hidden_dense/BiasAdd/ReadVariableOp#^hidden_dense/MatMul/ReadVariableOp*^hidden_dense_value/BiasAdd/ReadVariableOp)^hidden_dense_value/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????TT::::::::::::2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2P
&dense_advantage/BiasAdd/ReadVariableOp&dense_advantage/BiasAdd/ReadVariableOp2N
%dense_advantage/MatMul/ReadVariableOp%dense_advantage/MatMul/ReadVariableOp2J
#hidden_dense/BiasAdd/ReadVariableOp#hidden_dense/BiasAdd/ReadVariableOp2H
"hidden_dense/MatMul/ReadVariableOp"hidden_dense/MatMul/ReadVariableOp2V
)hidden_dense_value/BiasAdd/ReadVariableOp)hidden_dense_value/BiasAdd/ReadVariableOp2T
(hidden_dense_value/MatMul/ReadVariableOp(hidden_dense_value/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
3__inference_dense_advantage_layer_call_fn_220731111

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*W
fRRP
N__inference_dense_advantage_layer_call_and_return_conditional_losses_2207306882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?	
?
K__inference_hidden_dense_layer_call_and_return_conditional_losses_220730666

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
p
R__inference_tf_op_layer_Cast_14_layer_call_and_return_conditional_losses_220731049
inputs_0
identity|
Cast_14Castinputs_0*

DstT0*

SrcT0*
_cloned(*/
_output_shapes
:?????????TT2	
Cast_14g
IdentityIdentityCast_14:y:0*
T0*/
_output_shapes
:?????????TT2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????TT:( $
"
_user_specified_name
inputs/0
?
s
U__inference_tf_op_layer_truediv_14_layer_call_and_return_conditional_losses_220731060
inputs_0
identitya
truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
truediv_14/y?

truediv_14RealDivinputs_0truediv_14/y:output:0*
T0*
_cloned(*/
_output_shapes
:?????????TT2

truediv_14j
IdentityIdentitytruediv_14:z:0*
T0*/
_output_shapes
:?????????TT2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????TT:( $
"
_user_specified_name
inputs/0
?
z
P__inference_tf_op_layer_Sub_8_layer_call_and_return_conditional_losses_220730742

inputs
inputs_1
identityh
Sub_8Subinputsinputs_1*
T0*
_cloned(*'
_output_shapes
:?????????2
Sub_8]
IdentityIdentity	Sub_8:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
o
Q__inference_tf_op_layer_Mean_8_layer_call_and_return_conditional_losses_220731117
inputs_0
identityv
Mean_8/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean_8/reduction_indices?
Mean_8Meaninputs_0!Mean_8/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:?????????*
	keep_dims(2
Mean_8c
IdentityIdentityMean_8:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
?
N__inference_dense_advantage_layer_call_and_return_conditional_losses_220731104

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
z
P__inference_tf_op_layer_add_8_layer_call_and_return_conditional_losses_220730757

inputs
inputs_1
identityj
add_8AddV2inputsinputs_1*
T0*
_cloned(*'
_output_shapes
:?????????2
add_8]
IdentityIdentity	add_8:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?G
?
B__inference_dqn_layer_call_and_return_conditional_losses_220730957

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource/
+hidden_dense_matmul_readvariableop_resource0
,hidden_dense_biasadd_readvariableop_resource2
.dense_advantage_matmul_readvariableop_resource3
/dense_advantage_biasadd_readvariableop_resource5
1hidden_dense_value_matmul_readvariableop_resource6
2hidden_dense_value_biasadd_readvariableop_resource
identity??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?conv3/BiasAdd/ReadVariableOp?conv3/Conv2D/ReadVariableOp?&dense_advantage/BiasAdd/ReadVariableOp?%dense_advantage/MatMul/ReadVariableOp?#hidden_dense/BiasAdd/ReadVariableOp?"hidden_dense/MatMul/ReadVariableOp?)hidden_dense_value/BiasAdd/ReadVariableOp?(hidden_dense_value/MatMul/ReadVariableOp?
tf_op_layer_Cast_14/Cast_14Castinputs*

DstT0*

SrcT0*
_cloned(*/
_output_shapes
:?????????TT2
tf_op_layer_Cast_14/Cast_14?
#tf_op_layer_truediv_14/truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2%
#tf_op_layer_truediv_14/truediv_14/y?
!tf_op_layer_truediv_14/truediv_14RealDivtf_op_layer_Cast_14/Cast_14:y:0,tf_op_layer_truediv_14/truediv_14/y:output:0*
T0*
_cloned(*/
_output_shapes
:?????????TT2#
!tf_op_layer_truediv_14/truediv_14?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2D%tf_op_layer_truediv_14/truediv_14:z:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1/Conv2D?
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2

conv1/Relu?
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2/Conv2D/ReadVariableOp?
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 *
paddingVALID*
strides
2
conv2/Conv2D?
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2/BiasAdd/ReadVariableOp?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		 2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????		 2

conv2/Relu?
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv3/Conv2D/ReadVariableOp?
conv3/Conv2DConv2Dconv2/Relu:activations:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv3/Conv2D?
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv3/BiasAdd/ReadVariableOp?
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv3/BiasAddr

conv3/ReluReluconv3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2

conv3/Reluu
flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_14/Const?
flatten_14/ReshapeReshapeconv3/Relu:activations:0flatten_14/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_14/Reshape?
"hidden_dense/MatMul/ReadVariableOpReadVariableOp+hidden_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"hidden_dense/MatMul/ReadVariableOp?
hidden_dense/MatMulMatMulflatten_14/Reshape:output:0*hidden_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_dense/MatMul?
#hidden_dense/BiasAdd/ReadVariableOpReadVariableOp,hidden_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#hidden_dense/BiasAdd/ReadVariableOp?
hidden_dense/BiasAddBiasAddhidden_dense/MatMul:product:0+hidden_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_dense/BiasAdd?
hidden_dense/TanhTanhhidden_dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
hidden_dense/Tanh?
%dense_advantage/MatMul/ReadVariableOpReadVariableOp.dense_advantage_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%dense_advantage/MatMul/ReadVariableOp?
dense_advantage/MatMulMatMulhidden_dense/Tanh:y:0-dense_advantage/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_advantage/MatMul?
&dense_advantage/BiasAdd/ReadVariableOpReadVariableOp/dense_advantage_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&dense_advantage/BiasAdd/ReadVariableOp?
dense_advantage/BiasAddBiasAdd dense_advantage/MatMul:product:0.dense_advantage/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_advantage/BiasAdd?
+tf_op_layer_Mean_8/Mean_8/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2-
+tf_op_layer_Mean_8/Mean_8/reduction_indices?
tf_op_layer_Mean_8/Mean_8Mean dense_advantage/BiasAdd:output:04tf_op_layer_Mean_8/Mean_8/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:?????????*
	keep_dims(2
tf_op_layer_Mean_8/Mean_8?
(hidden_dense_value/MatMul/ReadVariableOpReadVariableOp1hidden_dense_value_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(hidden_dense_value/MatMul/ReadVariableOp?
hidden_dense_value/MatMulMatMulhidden_dense/Tanh:y:00hidden_dense_value/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
hidden_dense_value/MatMul?
)hidden_dense_value/BiasAdd/ReadVariableOpReadVariableOp2hidden_dense_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)hidden_dense_value/BiasAdd/ReadVariableOp?
hidden_dense_value/BiasAddBiasAdd#hidden_dense_value/MatMul:product:01hidden_dense_value/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
hidden_dense_value/BiasAdd?
tf_op_layer_Sub_8/Sub_8Sub dense_advantage/BiasAdd:output:0"tf_op_layer_Mean_8/Mean_8:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2
tf_op_layer_Sub_8/Sub_8?
tf_op_layer_add_8/add_8AddV2#hidden_dense_value/BiasAdd:output:0tf_op_layer_Sub_8/Sub_8:z:0*
T0*
_cloned(*'
_output_shapes
:?????????2
tf_op_layer_add_8/add_8?
IdentityIdentitytf_op_layer_add_8/add_8:z:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp'^dense_advantage/BiasAdd/ReadVariableOp&^dense_advantage/MatMul/ReadVariableOp$^hidden_dense/BiasAdd/ReadVariableOp#^hidden_dense/MatMul/ReadVariableOp*^hidden_dense_value/BiasAdd/ReadVariableOp)^hidden_dense_value/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????TT::::::::::::2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2P
&dense_advantage/BiasAdd/ReadVariableOp&dense_advantage/BiasAdd/ReadVariableOp2N
%dense_advantage/MatMul/ReadVariableOp%dense_advantage/MatMul/ReadVariableOp2J
#hidden_dense/BiasAdd/ReadVariableOp#hidden_dense/BiasAdd/ReadVariableOp2H
"hidden_dense/MatMul/ReadVariableOp"hidden_dense/MatMul/ReadVariableOp2V
)hidden_dense_value/BiasAdd/ReadVariableOp)hidden_dense_value/BiasAdd/ReadVariableOp2T
(hidden_dense_value/MatMul/ReadVariableOp(hidden_dense_value/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
U
7__inference_tf_op_layer_Cast_14_layer_call_fn_220731054
inputs_0
identity?
PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????TT*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*[
fVRT
R__inference_tf_op_layer_Cast_14_layer_call_and_return_conditional_losses_2207306102
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????TT2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????TT:( $
"
_user_specified_name
inputs/0
?
?
D__inference_conv1_layer_call_and_return_conditional_losses_220730551

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
|
P__inference_tf_op_layer_add_8_layer_call_and_return_conditional_losses_220731157
inputs_0
inputs_1
identityl
add_8AddV2inputs_0inputs_1*
T0*
_cloned(*'
_output_shapes
:?????????2
add_8]
IdentityIdentity	add_8:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?7
?
%__inference__traced_restore_220731271
file_prefix$
 assignvariableop_conv1_14_kernel$
 assignvariableop_1_conv1_14_bias&
"assignvariableop_2_conv2_14_kernel$
 assignvariableop_3_conv2_14_bias&
"assignvariableop_4_conv3_14_kernel$
 assignvariableop_5_conv3_14_bias-
)assignvariableop_6_hidden_dense_14_kernel+
'assignvariableop_7_hidden_dense_14_bias/
+assignvariableop_8_dense_advantage_8_kernel-
)assignvariableop_9_dense_advantage_8_bias4
0assignvariableop_10_hidden_dense_value_14_kernel2
.assignvariableop_11_hidden_dense_value_14_bias
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv1_14_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1_14_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2_14_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2_14_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv3_14_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv3_14_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp)assignvariableop_6_hidden_dense_14_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp'assignvariableop_7_hidden_dense_14_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp+assignvariableop_8_dense_advantage_8_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp)assignvariableop_9_dense_advantage_8_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp0assignvariableop_10_hidden_dense_value_14_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp.assignvariableop_11_hidden_dense_value_14_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12?
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_13"#
identity_13Identity_13:output:0*E
_input_shapes4
2: ::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
?
?
'__inference_signature_wrapper_220730904
state_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstate_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*-
f(R&
$__inference__wrapped_model_2207305382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????TT::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:+ '
%
_user_specified_namestate_input
?
?
D__inference_conv2_layer_call_and_return_conditional_losses_220730572

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
D__inference_conv3_layer_call_and_return_conditional_losses_220730593

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
?
e
I__inference_flatten_14_layer_call_and_return_conditional_losses_220731071

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
?
'__inference_dqn_layer_call_fn_220730886
state_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstate_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_dqn_layer_call_and_return_conditional_losses_2207308712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????TT::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:+ '
%
_user_specified_namestate_input
?8
?
B__inference_dqn_layer_call_and_return_conditional_losses_220730871

inputs(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2(
$conv2_statefulpartitionedcall_args_1(
$conv2_statefulpartitionedcall_args_2(
$conv3_statefulpartitionedcall_args_1(
$conv3_statefulpartitionedcall_args_2/
+hidden_dense_statefulpartitionedcall_args_1/
+hidden_dense_statefulpartitionedcall_args_22
.dense_advantage_statefulpartitionedcall_args_12
.dense_advantage_statefulpartitionedcall_args_25
1hidden_dense_value_statefulpartitionedcall_args_15
1hidden_dense_value_statefulpartitionedcall_args_2
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?'dense_advantage/StatefulPartitionedCall?$hidden_dense/StatefulPartitionedCall?*hidden_dense_value/StatefulPartitionedCall?
#tf_op_layer_Cast_14/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????TT*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*[
fVRT
R__inference_tf_op_layer_Cast_14_layer_call_and_return_conditional_losses_2207306102%
#tf_op_layer_Cast_14/PartitionedCall?
&tf_op_layer_truediv_14/PartitionedCallPartitionedCall,tf_op_layer_Cast_14/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????TT*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*^
fYRW
U__inference_tf_op_layer_truediv_14_layer_call_and_return_conditional_losses_2207306242(
&tf_op_layer_truediv_14/PartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_truediv_14/PartitionedCall:output:0$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_conv1_layer_call_and_return_conditional_losses_2207305512
conv1/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0$conv2_statefulpartitionedcall_args_1$conv2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????		 *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_conv2_layer_call_and_return_conditional_losses_2207305722
conv2/StatefulPartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0$conv3_statefulpartitionedcall_args_1$conv3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_conv3_layer_call_and_return_conditional_losses_2207305932
conv3/StatefulPartitionedCall?
flatten_14/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*R
fMRK
I__inference_flatten_14_layer_call_and_return_conditional_losses_2207306472
flatten_14/PartitionedCall?
$hidden_dense/StatefulPartitionedCallStatefulPartitionedCall#flatten_14/PartitionedCall:output:0+hidden_dense_statefulpartitionedcall_args_1+hidden_dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*T
fORM
K__inference_hidden_dense_layer_call_and_return_conditional_losses_2207306662&
$hidden_dense/StatefulPartitionedCall?
'dense_advantage/StatefulPartitionedCallStatefulPartitionedCall-hidden_dense/StatefulPartitionedCall:output:0.dense_advantage_statefulpartitionedcall_args_1.dense_advantage_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*W
fRRP
N__inference_dense_advantage_layer_call_and_return_conditional_losses_2207306882)
'dense_advantage/StatefulPartitionedCall?
"tf_op_layer_Mean_8/PartitionedCallPartitionedCall0dense_advantage/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Z
fURS
Q__inference_tf_op_layer_Mean_8_layer_call_and_return_conditional_losses_2207307062$
"tf_op_layer_Mean_8/PartitionedCall?
*hidden_dense_value/StatefulPartitionedCallStatefulPartitionedCall-hidden_dense/StatefulPartitionedCall:output:01hidden_dense_value_statefulpartitionedcall_args_11hidden_dense_value_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Z
fURS
Q__inference_hidden_dense_value_layer_call_and_return_conditional_losses_2207307242,
*hidden_dense_value/StatefulPartitionedCall?
!tf_op_layer_Sub_8/PartitionedCallPartitionedCall0dense_advantage/StatefulPartitionedCall:output:0+tf_op_layer_Mean_8/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Y
fTRR
P__inference_tf_op_layer_Sub_8_layer_call_and_return_conditional_losses_2207307422#
!tf_op_layer_Sub_8/PartitionedCall?
!tf_op_layer_add_8/PartitionedCallPartitionedCall3hidden_dense_value/StatefulPartitionedCall:output:0*tf_op_layer_Sub_8/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Y
fTRR
P__inference_tf_op_layer_add_8_layer_call_and_return_conditional_losses_2207307572#
!tf_op_layer_add_8/PartitionedCall?
IdentityIdentity*tf_op_layer_add_8/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall(^dense_advantage/StatefulPartitionedCall%^hidden_dense/StatefulPartitionedCall+^hidden_dense_value/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????TT::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2R
'dense_advantage/StatefulPartitionedCall'dense_advantage/StatefulPartitionedCall2L
$hidden_dense/StatefulPartitionedCall$hidden_dense/StatefulPartitionedCall2X
*hidden_dense_value/StatefulPartitionedCall*hidden_dense_value/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
a
5__inference_tf_op_layer_Sub_8_layer_call_fn_220731151
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Y
fTRR
P__inference_tf_op_layer_Sub_8_layer_call_and_return_conditional_losses_2207307422
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
?
6__inference_hidden_dense_value_layer_call_fn_220731139

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Z
fURS
Q__inference_hidden_dense_value_layer_call_and_return_conditional_losses_2207307242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
|
P__inference_tf_op_layer_Sub_8_layer_call_and_return_conditional_losses_220731145
inputs_0
inputs_1
identityj
Sub_8Subinputs_0inputs_1*
T0*
_cloned(*'
_output_shapes
:?????????2
Sub_8]
IdentityIdentity	Sub_8:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?	
?
K__inference_hidden_dense_layer_call_and_return_conditional_losses_220731087

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
T
6__inference_tf_op_layer_Mean_8_layer_call_fn_220731122
inputs_0
identity?
PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Z
fURS
Q__inference_tf_op_layer_Mean_8_layer_call_and_return_conditional_losses_2207307062
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:( $
"
_user_specified_name
inputs/0
?
n
R__inference_tf_op_layer_Cast_14_layer_call_and_return_conditional_losses_220730610

inputs
identityz
Cast_14Castinputs*

DstT0*

SrcT0*
_cloned(*/
_output_shapes
:?????????TT2	
Cast_14g
IdentityIdentityCast_14:y:0*
T0*/
_output_shapes
:?????????TT2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????TT:& "
 
_user_specified_nameinputs
?8
?
B__inference_dqn_layer_call_and_return_conditional_losses_220730767
state_input(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2(
$conv2_statefulpartitionedcall_args_1(
$conv2_statefulpartitionedcall_args_2(
$conv3_statefulpartitionedcall_args_1(
$conv3_statefulpartitionedcall_args_2/
+hidden_dense_statefulpartitionedcall_args_1/
+hidden_dense_statefulpartitionedcall_args_22
.dense_advantage_statefulpartitionedcall_args_12
.dense_advantage_statefulpartitionedcall_args_25
1hidden_dense_value_statefulpartitionedcall_args_15
1hidden_dense_value_statefulpartitionedcall_args_2
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?'dense_advantage/StatefulPartitionedCall?$hidden_dense/StatefulPartitionedCall?*hidden_dense_value/StatefulPartitionedCall?
#tf_op_layer_Cast_14/PartitionedCallPartitionedCallstate_input*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????TT*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*[
fVRT
R__inference_tf_op_layer_Cast_14_layer_call_and_return_conditional_losses_2207306102%
#tf_op_layer_Cast_14/PartitionedCall?
&tf_op_layer_truediv_14/PartitionedCallPartitionedCall,tf_op_layer_Cast_14/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????TT*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*^
fYRW
U__inference_tf_op_layer_truediv_14_layer_call_and_return_conditional_losses_2207306242(
&tf_op_layer_truediv_14/PartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_truediv_14/PartitionedCall:output:0$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_conv1_layer_call_and_return_conditional_losses_2207305512
conv1/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0$conv2_statefulpartitionedcall_args_1$conv2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????		 *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_conv2_layer_call_and_return_conditional_losses_2207305722
conv2/StatefulPartitionedCall?
conv3/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0$conv3_statefulpartitionedcall_args_1$conv3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? *3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_conv3_layer_call_and_return_conditional_losses_2207305932
conv3/StatefulPartitionedCall?
flatten_14/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*R
fMRK
I__inference_flatten_14_layer_call_and_return_conditional_losses_2207306472
flatten_14/PartitionedCall?
$hidden_dense/StatefulPartitionedCallStatefulPartitionedCall#flatten_14/PartitionedCall:output:0+hidden_dense_statefulpartitionedcall_args_1+hidden_dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*T
fORM
K__inference_hidden_dense_layer_call_and_return_conditional_losses_2207306662&
$hidden_dense/StatefulPartitionedCall?
'dense_advantage/StatefulPartitionedCallStatefulPartitionedCall-hidden_dense/StatefulPartitionedCall:output:0.dense_advantage_statefulpartitionedcall_args_1.dense_advantage_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*W
fRRP
N__inference_dense_advantage_layer_call_and_return_conditional_losses_2207306882)
'dense_advantage/StatefulPartitionedCall?
"tf_op_layer_Mean_8/PartitionedCallPartitionedCall0dense_advantage/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Z
fURS
Q__inference_tf_op_layer_Mean_8_layer_call_and_return_conditional_losses_2207307062$
"tf_op_layer_Mean_8/PartitionedCall?
*hidden_dense_value/StatefulPartitionedCallStatefulPartitionedCall-hidden_dense/StatefulPartitionedCall:output:01hidden_dense_value_statefulpartitionedcall_args_11hidden_dense_value_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Z
fURS
Q__inference_hidden_dense_value_layer_call_and_return_conditional_losses_2207307242,
*hidden_dense_value/StatefulPartitionedCall?
!tf_op_layer_Sub_8/PartitionedCallPartitionedCall0dense_advantage/StatefulPartitionedCall:output:0+tf_op_layer_Mean_8/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Y
fTRR
P__inference_tf_op_layer_Sub_8_layer_call_and_return_conditional_losses_2207307422#
!tf_op_layer_Sub_8/PartitionedCall?
!tf_op_layer_add_8/PartitionedCallPartitionedCall3hidden_dense_value/StatefulPartitionedCall:output:0*tf_op_layer_Sub_8/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*Y
fTRR
P__inference_tf_op_layer_add_8_layer_call_and_return_conditional_losses_2207307572#
!tf_op_layer_add_8/PartitionedCall?
IdentityIdentity*tf_op_layer_add_8/PartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall(^dense_advantage/StatefulPartitionedCall%^hidden_dense/StatefulPartitionedCall+^hidden_dense_value/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????TT::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2R
'dense_advantage/StatefulPartitionedCall'dense_advantage/StatefulPartitionedCall2L
$hidden_dense/StatefulPartitionedCall$hidden_dense/StatefulPartitionedCall2X
*hidden_dense_value/StatefulPartitionedCall*hidden_dense_value/StatefulPartitionedCall:+ '
%
_user_specified_namestate_input
?
e
I__inference_flatten_14_layer_call_and_return_conditional_losses_220730647

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
?
'__inference_dqn_layer_call_fn_220731044

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*K
fFRD
B__inference_dqn_layer_call_and_return_conditional_losses_2207308712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????TT::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
q
U__inference_tf_op_layer_truediv_14_layer_call_and_return_conditional_losses_220730624

inputs
identitya
truediv_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
truediv_14/y?

truediv_14RealDivinputstruediv_14/y:output:0*
T0*
_cloned(*/
_output_shapes
:?????????TT2

truediv_14j
IdentityIdentitytruediv_14:z:0*
T0*/
_output_shapes
:?????????TT2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????TT:& "
 
_user_specified_nameinputs
?&
?
"__inference__traced_save_220731223
file_prefix.
*savev2_conv1_14_kernel_read_readvariableop,
(savev2_conv1_14_bias_read_readvariableop.
*savev2_conv2_14_kernel_read_readvariableop,
(savev2_conv2_14_bias_read_readvariableop.
*savev2_conv3_14_kernel_read_readvariableop,
(savev2_conv3_14_bias_read_readvariableop5
1savev2_hidden_dense_14_kernel_read_readvariableop3
/savev2_hidden_dense_14_bias_read_readvariableop7
3savev2_dense_advantage_8_kernel_read_readvariableop5
1savev2_dense_advantage_8_bias_read_readvariableop;
7savev2_hidden_dense_value_14_kernel_read_readvariableop9
5savev2_hidden_dense_value_14_bias_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_a0c84b6ab28242cd82dfbfd8afe61128/part2
StringJoin/inputs_1?

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1_14_kernel_read_readvariableop(savev2_conv1_14_bias_read_readvariableop*savev2_conv2_14_kernel_read_readvariableop(savev2_conv2_14_bias_read_readvariableop*savev2_conv3_14_kernel_read_readvariableop(savev2_conv3_14_bias_read_readvariableop1savev2_hidden_dense_14_kernel_read_readvariableop/savev2_hidden_dense_14_bias_read_readvariableop3savev2_dense_advantage_8_kernel_read_readvariableop1savev2_dense_advantage_8_bias_read_readvariableop7savev2_hidden_dense_value_14_kernel_read_readvariableop5savev2_hidden_dense_value_14_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::: : :  : :
??:?:	?::	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
?
?
)__inference_conv1_layer_call_fn_220730559

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*M
fHRF
D__inference_conv1_layer_call_and_return_conditional_losses_2207305512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
K
state_input<
serving_default_state_input:0?????????TTE
tf_op_layer_add_80
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?`
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
	variables
regularization_losses
trainable_variables
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?\
_tf_keras_model?\{"class_name": "Model", "name": "dqn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dqn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 84, 84, 4], "dtype": "uint8", "sparse": false, "ragged": false, "name": "state_input"}, "name": "state_input", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Cast_14", "trainable": true, "dtype": "float32", "node_def": {"name": "Cast_14", "op": "Cast", "input": ["state_input_14"], "attr": {"SrcT": {"type": "DT_UINT8"}, "Truncate": {"b": false}, "DstT": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Cast_14", "inbound_nodes": [[["state_input", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "truediv_14", "trainable": true, "dtype": "float32", "node_def": {"name": "truediv_14", "op": "RealDiv", "input": ["Cast_14", "truediv_14/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 255.0}}, "name": "tf_op_layer_truediv_14", "inbound_nodes": [[["tf_op_layer_Cast_14", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [8, 8], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["tf_op_layer_truediv_14", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [4, 4], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_14", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_14", "inbound_nodes": [[["conv3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_dense", "inbound_nodes": [[["flatten_14", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_advantage", "trainable": true, "dtype": "float32", "units": 18, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_advantage", "inbound_nodes": [[["hidden_dense", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mean_8", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean_8", "op": "Mean", "input": ["dense_advantage_8/Identity", "Mean_8/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_Mean_8", "inbound_nodes": [[["dense_advantage", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_dense_value", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_dense_value", "inbound_nodes": [[["hidden_dense", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub_8", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub_8", "op": "Sub", "input": ["dense_advantage_8/Identity", "Mean_8"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Sub_8", "inbound_nodes": [[["dense_advantage", 0, 0, {}], ["tf_op_layer_Mean_8", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "add_8", "trainable": true, "dtype": "float32", "node_def": {"name": "add_8", "op": "AddV2", "input": ["hidden_dense_value_14/Identity", "Sub_8"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_add_8", "inbound_nodes": [[["hidden_dense_value", 0, 0, {}], ["tf_op_layer_Sub_8", 0, 0, {}]]]}], "input_layers": [["state_input", 0, 0]], "output_layers": [["tf_op_layer_add_8", 0, 0]]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "dqn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 84, 84, 4], "dtype": "uint8", "sparse": false, "ragged": false, "name": "state_input"}, "name": "state_input", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Cast_14", "trainable": true, "dtype": "float32", "node_def": {"name": "Cast_14", "op": "Cast", "input": ["state_input_14"], "attr": {"SrcT": {"type": "DT_UINT8"}, "Truncate": {"b": false}, "DstT": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Cast_14", "inbound_nodes": [[["state_input", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "truediv_14", "trainable": true, "dtype": "float32", "node_def": {"name": "truediv_14", "op": "RealDiv", "input": ["Cast_14", "truediv_14/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 255.0}}, "name": "tf_op_layer_truediv_14", "inbound_nodes": [[["tf_op_layer_Cast_14", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [8, 8], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["tf_op_layer_truediv_14", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [4, 4], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_14", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_14", "inbound_nodes": [[["conv3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_dense", "inbound_nodes": [[["flatten_14", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_advantage", "trainable": true, "dtype": "float32", "units": 18, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_advantage", "inbound_nodes": [[["hidden_dense", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mean_8", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean_8", "op": "Mean", "input": ["dense_advantage_8/Identity", "Mean_8/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}, "name": "tf_op_layer_Mean_8", "inbound_nodes": [[["dense_advantage", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_dense_value", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_dense_value", "inbound_nodes": [[["hidden_dense", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub_8", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub_8", "op": "Sub", "input": ["dense_advantage_8/Identity", "Mean_8"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Sub_8", "inbound_nodes": [[["dense_advantage", 0, 0, {}], ["tf_op_layer_Mean_8", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "add_8", "trainable": true, "dtype": "float32", "node_def": {"name": "add_8", "op": "AddV2", "input": ["hidden_dense_value_14/Identity", "Sub_8"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_add_8", "inbound_nodes": [[["hidden_dense_value", 0, 0, {}], ["tf_op_layer_Sub_8", 0, 0, {}]]]}], "input_layers": [["state_input", 0, 0]], "output_layers": [["tf_op_layer_add_8", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "state_input", "dtype": "uint8", "sparse": false, "ragged": false, "batch_input_shape": [null, 84, 84, 4], "config": {"batch_input_shape": [null, 84, 84, 4], "dtype": "uint8", "sparse": false, "ragged": false, "name": "state_input"}}
?
	constants
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Cast_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Cast_14", "trainable": true, "dtype": "float32", "node_def": {"name": "Cast_14", "op": "Cast", "input": ["state_input_14"], "attr": {"SrcT": {"type": "DT_UINT8"}, "Truncate": {"b": false}, "DstT": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
	constants
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_truediv_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "truediv_14", "trainable": true, "dtype": "float32", "node_def": {"name": "truediv_14", "op": "RealDiv", "input": ["Cast_14", "truediv_14/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 255.0}}}
?

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [8, 8], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}}}
?

#kernel
$bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [4, 4], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}}
?

)kernel
*bias
+	variables
,regularization_losses
-trainable_variables
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
?
/	variables
0regularization_losses
1trainable_variables
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_14", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

3kernel
4bias
5	variables
6regularization_losses
7trainable_variables
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden_dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1568}}}}
?

9kernel
:bias
;	variables
<regularization_losses
=trainable_variables
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_advantage", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_advantage", "trainable": true, "dtype": "float32", "units": 18, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}}
?
?	constants
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mean_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Mean_8", "trainable": true, "dtype": "float32", "node_def": {"name": "Mean_8", "op": "Mean", "input": ["dense_advantage_8/Identity", "Mean_8/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": true}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1}}}
?

Dkernel
Ebias
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden_dense_value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_dense_value", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}}
?
J	constants
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Sub_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Sub_8", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub_8", "op": "Sub", "input": ["dense_advantage_8/Identity", "Mean_8"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
O	constants
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_add_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_8", "trainable": true, "dtype": "float32", "node_def": {"name": "add_8", "op": "AddV2", "input": ["hidden_dense_value_14/Identity", "Sub_8"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
v
0
1
#2
$3
)4
*5
36
47
98
:9
D10
E11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
#2
$3
)4
*5
36
47
98
:9
D10
E11"
trackable_list_wrapper
?
	variables
regularization_losses
Tmetrics
Unon_trainable_variables

Vlayers
Wlayer_regularization_losses
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
regularization_losses
Xmetrics
Ynon_trainable_variables
Zlayer_regularization_losses

[layers
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
regularization_losses
\metrics
]non_trainable_variables
^layer_regularization_losses

_layers
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2conv1_14/kernel
:2conv1_14/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
 regularization_losses
`metrics
anon_trainable_variables
blayer_regularization_losses

clayers
!trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2_14/kernel
: 2conv2_14/bias
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?
%	variables
&regularization_losses
dmetrics
enon_trainable_variables
flayer_regularization_losses

glayers
'trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'  2conv3_14/kernel
: 2conv3_14/bias
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
+	variables
,regularization_losses
hmetrics
inon_trainable_variables
jlayer_regularization_losses

klayers
-trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
/	variables
0regularization_losses
lmetrics
mnon_trainable_variables
nlayer_regularization_losses

olayers
1trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(
??2hidden_dense_14/kernel
#:!?2hidden_dense_14/bias
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
5	variables
6regularization_losses
pmetrics
qnon_trainable_variables
rlayer_regularization_losses

slayers
7trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)	?2dense_advantage_8/kernel
$:"2dense_advantage_8/bias
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?
;	variables
<regularization_losses
tmetrics
unon_trainable_variables
vlayer_regularization_losses

wlayers
=trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
@	variables
Aregularization_losses
xmetrics
ynon_trainable_variables
zlayer_regularization_losses

{layers
Btrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-	?2hidden_dense_value_14/kernel
(:&2hidden_dense_value_14/bias
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
?
F	variables
Gregularization_losses
|metrics
}non_trainable_variables
~layer_regularization_losses

layers
Htrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
K	variables
Lregularization_losses
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?layers
Mtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
P	variables
Qregularization_losses
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?layers
Rtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
'__inference_dqn_layer_call_fn_220731044
'__inference_dqn_layer_call_fn_220731027
'__inference_dqn_layer_call_fn_220730841
'__inference_dqn_layer_call_fn_220730886?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dqn_layer_call_and_return_conditional_losses_220730957
B__inference_dqn_layer_call_and_return_conditional_losses_220731010
B__inference_dqn_layer_call_and_return_conditional_losses_220730795
B__inference_dqn_layer_call_and_return_conditional_losses_220730767?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference__wrapped_model_220730538?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *2?/
-?*
state_input?????????TT
?2?
7__inference_tf_op_layer_Cast_14_layer_call_fn_220731054?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_tf_op_layer_Cast_14_layer_call_and_return_conditional_losses_220731049?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
:__inference_tf_op_layer_truediv_14_layer_call_fn_220731065?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
U__inference_tf_op_layer_truediv_14_layer_call_and_return_conditional_losses_220731060?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv1_layer_call_fn_220730559?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
D__inference_conv1_layer_call_and_return_conditional_losses_220730551?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
)__inference_conv2_layer_call_fn_220730580?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
D__inference_conv2_layer_call_and_return_conditional_losses_220730572?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
)__inference_conv3_layer_call_fn_220730601?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
D__inference_conv3_layer_call_and_return_conditional_losses_220730593?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
.__inference_flatten_14_layer_call_fn_220731076?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_flatten_14_layer_call_and_return_conditional_losses_220731071?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_hidden_dense_layer_call_fn_220731094?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_hidden_dense_layer_call_and_return_conditional_losses_220731087?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_dense_advantage_layer_call_fn_220731111?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_dense_advantage_layer_call_and_return_conditional_losses_220731104?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_tf_op_layer_Mean_8_layer_call_fn_220731122?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_tf_op_layer_Mean_8_layer_call_and_return_conditional_losses_220731117?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_hidden_dense_value_layer_call_fn_220731139?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_hidden_dense_value_layer_call_and_return_conditional_losses_220731132?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_tf_op_layer_Sub_8_layer_call_fn_220731151?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_tf_op_layer_Sub_8_layer_call_and_return_conditional_losses_220731145?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_tf_op_layer_add_8_layer_call_fn_220731163?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_tf_op_layer_add_8_layer_call_and_return_conditional_losses_220731157?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:B8
'__inference_signature_wrapper_220730904state_input?
$__inference__wrapped_model_220730538?#$)*349:DE<?9
2?/
-?*
state_input?????????TT
? "E?B
@
tf_op_layer_add_8+?(
tf_op_layer_add_8??????????
D__inference_conv1_layer_call_and_return_conditional_losses_220730551?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
)__inference_conv1_layer_call_fn_220730559?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
D__inference_conv2_layer_call_and_return_conditional_losses_220730572?#$I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+??????????????????????????? 
? ?
)__inference_conv2_layer_call_fn_220730580?#$I?F
??<
:?7
inputs+???????????????????????????
? "2?/+??????????????????????????? ?
D__inference_conv3_layer_call_and_return_conditional_losses_220730593?)*I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
)__inference_conv3_layer_call_fn_220730601?)*I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
N__inference_dense_advantage_layer_call_and_return_conditional_losses_220731104]9:0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
3__inference_dense_advantage_layer_call_fn_220731111P9:0?-
&?#
!?
inputs??????????
? "???????????
B__inference_dqn_layer_call_and_return_conditional_losses_220730767{#$)*349:DED?A
:?7
-?*
state_input?????????TT
p

 
? "%?"
?
0?????????
? ?
B__inference_dqn_layer_call_and_return_conditional_losses_220730795{#$)*349:DED?A
:?7
-?*
state_input?????????TT
p 

 
? "%?"
?
0?????????
? ?
B__inference_dqn_layer_call_and_return_conditional_losses_220730957v#$)*349:DE??<
5?2
(?%
inputs?????????TT
p

 
? "%?"
?
0?????????
? ?
B__inference_dqn_layer_call_and_return_conditional_losses_220731010v#$)*349:DE??<
5?2
(?%
inputs?????????TT
p 

 
? "%?"
?
0?????????
? ?
'__inference_dqn_layer_call_fn_220730841n#$)*349:DED?A
:?7
-?*
state_input?????????TT
p

 
? "???????????
'__inference_dqn_layer_call_fn_220730886n#$)*349:DED?A
:?7
-?*
state_input?????????TT
p 

 
? "???????????
'__inference_dqn_layer_call_fn_220731027i#$)*349:DE??<
5?2
(?%
inputs?????????TT
p

 
? "???????????
'__inference_dqn_layer_call_fn_220731044i#$)*349:DE??<
5?2
(?%
inputs?????????TT
p 

 
? "???????????
I__inference_flatten_14_layer_call_and_return_conditional_losses_220731071a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????
? ?
.__inference_flatten_14_layer_call_fn_220731076T7?4
-?*
(?%
inputs????????? 
? "????????????
K__inference_hidden_dense_layer_call_and_return_conditional_losses_220731087^340?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
0__inference_hidden_dense_layer_call_fn_220731094Q340?-
&?#
!?
inputs??????????
? "????????????
Q__inference_hidden_dense_value_layer_call_and_return_conditional_losses_220731132]DE0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
6__inference_hidden_dense_value_layer_call_fn_220731139PDE0?-
&?#
!?
inputs??????????
? "???????????
'__inference_signature_wrapper_220730904?#$)*349:DEK?H
? 
A?>
<
state_input-?*
state_input?????????TT"E?B
@
tf_op_layer_add_8+?(
tf_op_layer_add_8??????????
R__inference_tf_op_layer_Cast_14_layer_call_and_return_conditional_losses_220731049o>?;
4?1
/?,
*?'
inputs/0?????????TT
? "-?*
#? 
0?????????TT
? ?
7__inference_tf_op_layer_Cast_14_layer_call_fn_220731054b>?;
4?1
/?,
*?'
inputs/0?????????TT
? " ??????????TT?
Q__inference_tf_op_layer_Mean_8_layer_call_and_return_conditional_losses_220731117_6?3
,?)
'?$
"?
inputs/0?????????
? "%?"
?
0?????????
? ?
6__inference_tf_op_layer_Mean_8_layer_call_fn_220731122R6?3
,?)
'?$
"?
inputs/0?????????
? "???????????
P__inference_tf_op_layer_Sub_8_layer_call_and_return_conditional_losses_220731145?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
5__inference_tf_op_layer_Sub_8_layer_call_fn_220731151vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
P__inference_tf_op_layer_add_8_layer_call_and_return_conditional_losses_220731157?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
5__inference_tf_op_layer_add_8_layer_call_fn_220731163vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
U__inference_tf_op_layer_truediv_14_layer_call_and_return_conditional_losses_220731060o>?;
4?1
/?,
*?'
inputs/0?????????TT
? "-?*
#? 
0?????????TT
? ?
:__inference_tf_op_layer_truediv_14_layer_call_fn_220731065b>?;
4?1
/?,
*?'
inputs/0?????????TT
? " ??????????TT