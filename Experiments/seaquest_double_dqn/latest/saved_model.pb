??
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
shapeshape?"serve*2.1.02unknown8??
?
conv1_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv1_16/kernel
{
#conv1_16/kernel/Read/ReadVariableOpReadVariableOpconv1_16/kernel*&
_output_shapes
:*
dtype0
r
conv1_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1_16/bias
k
!conv1_16/bias/Read/ReadVariableOpReadVariableOpconv1_16/bias*
_output_shapes
:*
dtype0
?
conv2_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2_16/kernel
{
#conv2_16/kernel/Read/ReadVariableOpReadVariableOpconv2_16/kernel*&
_output_shapes
: *
dtype0
r
conv2_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2_16/bias
k
!conv2_16/bias/Read/ReadVariableOpReadVariableOpconv2_16/bias*
_output_shapes
: *
dtype0
?
conv3_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv3_16/kernel
{
#conv3_16/kernel/Read/ReadVariableOpReadVariableOpconv3_16/kernel*&
_output_shapes
:  *
dtype0
r
conv3_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3_16/bias
k
!conv3_16/bias/Read/ReadVariableOpReadVariableOpconv3_16/bias*
_output_shapes
: *
dtype0
?
hidden_dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_namehidden_dense_16/kernel
?
*hidden_dense_16/kernel/Read/ReadVariableOpReadVariableOphidden_dense_16/kernel* 
_output_shapes
:
??*
dtype0
?
hidden_dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_namehidden_dense_16/bias
z
(hidden_dense_16/bias/Read/ReadVariableOpReadVariableOphidden_dense_16/bias*
_output_shapes	
:?*
dtype0
?
hidden_dense_value_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*-
shared_namehidden_dense_value_16/kernel
?
0hidden_dense_value_16/kernel/Read/ReadVariableOpReadVariableOphidden_dense_value_16/kernel* 
_output_shapes
:
??*
dtype0
?
hidden_dense_value_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namehidden_dense_value_16/bias
?
.hidden_dense_value_16/bias/Read/ReadVariableOpReadVariableOphidden_dense_value_16/bias*
_output_shapes	
:?*
dtype0
?
value_output_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_namevalue_output_6/kernel
?
)value_output_6/kernel/Read/ReadVariableOpReadVariableOpvalue_output_6/kernel*
_output_shapes
:	?*
dtype0
~
value_output_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namevalue_output_6/bias
w
'value_output_6/bias/Read/ReadVariableOpReadVariableOpvalue_output_6/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?#
value?#B?# B?#
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

layer_with_weights-5

layer-9
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
a
	constants
	variables
regularization_losses
trainable_variables
	keras_api
a
	constants
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

 kernel
!bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
h

&kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
R
,	variables
-regularization_losses
.trainable_variables
/	keras_api
h

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
h

6kernel
7bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api
h

<kernel
=bias
>	variables
?regularization_losses
@trainable_variables
A	keras_api
V
0
1
 2
!3
&4
'5
06
17
68
79
<10
=11
 
V
0
1
 2
!3
&4
'5
06
17
68
79
<10
=11
?
	variables
regularization_losses
Bmetrics
Cnon_trainable_variables

Dlayers
Elayer_regularization_losses
trainable_variables
 
 
 
 
 
?
	variables
regularization_losses
Fmetrics
Gnon_trainable_variables
Hlayer_regularization_losses

Ilayers
trainable_variables
 
 
 
 
?
	variables
regularization_losses
Jmetrics
Knon_trainable_variables
Llayer_regularization_losses

Mlayers
trainable_variables
[Y
VARIABLE_VALUEconv1_16/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1_16/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
regularization_losses
Nmetrics
Onon_trainable_variables
Player_regularization_losses

Qlayers
trainable_variables
[Y
VARIABLE_VALUEconv2_16/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2_16/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
?
"	variables
#regularization_losses
Rmetrics
Snon_trainable_variables
Tlayer_regularization_losses

Ulayers
$trainable_variables
[Y
VARIABLE_VALUEconv3_16/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3_16/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1
 

&0
'1
?
(	variables
)regularization_losses
Vmetrics
Wnon_trainable_variables
Xlayer_regularization_losses

Ylayers
*trainable_variables
 
 
 
?
,	variables
-regularization_losses
Zmetrics
[non_trainable_variables
\layer_regularization_losses

]layers
.trainable_variables
b`
VARIABLE_VALUEhidden_dense_16/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEhidden_dense_16/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
?
2	variables
3regularization_losses
^metrics
_non_trainable_variables
`layer_regularization_losses

alayers
4trainable_variables
hf
VARIABLE_VALUEhidden_dense_value_16/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEhidden_dense_value_16/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71
 

60
71
?
8	variables
9regularization_losses
bmetrics
cnon_trainable_variables
dlayer_regularization_losses

elayers
:trainable_variables
a_
VARIABLE_VALUEvalue_output_6/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEvalue_output_6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

<0
=1
 

<0
=1
?
>	variables
?regularization_losses
fmetrics
gnon_trainable_variables
hlayer_regularization_losses

ilayers
@trainable_variables
 
 
F
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_state_inputconv1_16/kernelconv1_16/biasconv2_16/kernelconv2_16/biasconv3_16/kernelconv3_16/biashidden_dense_16/kernelhidden_dense_16/biashidden_dense_value_16/kernelhidden_dense_value_16/biasvalue_output_6/kernelvalue_output_6/bias*
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
'__inference_signature_wrapper_330371475
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1_16/kernel/Read/ReadVariableOp!conv1_16/bias/Read/ReadVariableOp#conv2_16/kernel/Read/ReadVariableOp!conv2_16/bias/Read/ReadVariableOp#conv3_16/kernel/Read/ReadVariableOp!conv3_16/bias/Read/ReadVariableOp*hidden_dense_16/kernel/Read/ReadVariableOp(hidden_dense_16/bias/Read/ReadVariableOp0hidden_dense_value_16/kernel/Read/ReadVariableOp.hidden_dense_value_16/bias/Read/ReadVariableOp)value_output_6/kernel/Read/ReadVariableOp'value_output_6/bias/Read/ReadVariableOpConst*
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
"__inference__traced_save_330371754
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1_16/kernelconv1_16/biasconv2_16/kernelconv2_16/biasconv3_16/kernelconv3_16/biashidden_dense_16/kernelhidden_dense_16/biashidden_dense_value_16/kernelhidden_dense_value_16/biasvalue_output_6/kernelvalue_output_6/bias*
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
%__inference__traced_restore_330371802??
?	
?
Q__inference_hidden_dense_value_layer_call_and_return_conditional_losses_330371312

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
K__inference_value_output_layer_call_and_return_conditional_losses_330371334

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
'__inference_dqn_layer_call_fn_330371415
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
B__inference_dqn_layer_call_and_return_conditional_losses_3303714002
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
?
?
)__inference_conv3_layer_call_fn_330371224

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
D__inference_conv3_layer_call_and_return_conditional_losses_3303712162
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
?
?
'__inference_dqn_layer_call_fn_330371609

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
B__inference_dqn_layer_call_and_return_conditional_losses_3303714422
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
J
.__inference_flatten_16_layer_call_fn_330371641

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
I__inference_flatten_16_layer_call_and_return_conditional_losses_3303712702
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
?
?
'__inference_dqn_layer_call_fn_330371592

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
B__inference_dqn_layer_call_and_return_conditional_losses_3303714002
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
?A
?
B__inference_dqn_layer_call_and_return_conditional_losses_330371525

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource/
+hidden_dense_matmul_readvariableop_resource0
,hidden_dense_biasadd_readvariableop_resource5
1hidden_dense_value_matmul_readvariableop_resource6
2hidden_dense_value_biasadd_readvariableop_resource/
+value_output_matmul_readvariableop_resource0
,value_output_biasadd_readvariableop_resource
identity??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?conv3/BiasAdd/ReadVariableOp?conv3/Conv2D/ReadVariableOp?#hidden_dense/BiasAdd/ReadVariableOp?"hidden_dense/MatMul/ReadVariableOp?)hidden_dense_value/BiasAdd/ReadVariableOp?(hidden_dense_value/MatMul/ReadVariableOp?#value_output/BiasAdd/ReadVariableOp?"value_output/MatMul/ReadVariableOp?
tf_op_layer_Cast_16/Cast_16Castinputs*

DstT0*

SrcT0*
_cloned(*/
_output_shapes
:?????????TT2
tf_op_layer_Cast_16/Cast_16?
#tf_op_layer_truediv_16/truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2%
#tf_op_layer_truediv_16/truediv_16/y?
!tf_op_layer_truediv_16/truediv_16RealDivtf_op_layer_Cast_16/Cast_16:y:0,tf_op_layer_truediv_16/truediv_16/y:output:0*
T0*
_cloned(*/
_output_shapes
:?????????TT2#
!tf_op_layer_truediv_16/truediv_16?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2D%tf_op_layer_truediv_16/truediv_16:z:0#conv1/Conv2D/ReadVariableOp:value:0*
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
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_16/Const?
flatten_16/ReshapeReshapeconv3/Relu:activations:0flatten_16/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_16/Reshape?
"hidden_dense/MatMul/ReadVariableOpReadVariableOp+hidden_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"hidden_dense/MatMul/ReadVariableOp?
hidden_dense/MatMulMatMulflatten_16/Reshape:output:0*hidden_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_dense/MatMul?
#hidden_dense/BiasAdd/ReadVariableOpReadVariableOp,hidden_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#hidden_dense/BiasAdd/ReadVariableOp?
hidden_dense/BiasAddBiasAddhidden_dense/MatMul:product:0+hidden_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_dense/BiasAdd?
hidden_dense/TanhTanhhidden_dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
hidden_dense/Tanh?
(hidden_dense_value/MatMul/ReadVariableOpReadVariableOp1hidden_dense_value_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(hidden_dense_value/MatMul/ReadVariableOp?
hidden_dense_value/MatMulMatMulhidden_dense/Tanh:y:00hidden_dense_value/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_dense_value/MatMul?
)hidden_dense_value/BiasAdd/ReadVariableOpReadVariableOp2hidden_dense_value_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)hidden_dense_value/BiasAdd/ReadVariableOp?
hidden_dense_value/BiasAddBiasAdd#hidden_dense_value/MatMul:product:01hidden_dense_value/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_dense_value/BiasAdd?
hidden_dense_value/TanhTanh#hidden_dense_value/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
hidden_dense_value/Tanh?
"value_output/MatMul/ReadVariableOpReadVariableOp+value_output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"value_output/MatMul/ReadVariableOp?
value_output/MatMulMatMulhidden_dense_value/Tanh:y:0*value_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
value_output/MatMul?
#value_output/BiasAdd/ReadVariableOpReadVariableOp,value_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#value_output/BiasAdd/ReadVariableOp?
value_output/BiasAddBiasAddvalue_output/MatMul:product:0+value_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
value_output/BiasAdd?
IdentityIdentityvalue_output/BiasAdd:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp$^hidden_dense/BiasAdd/ReadVariableOp#^hidden_dense/MatMul/ReadVariableOp*^hidden_dense_value/BiasAdd/ReadVariableOp)^hidden_dense_value/MatMul/ReadVariableOp$^value_output/BiasAdd/ReadVariableOp#^value_output/MatMul/ReadVariableOp*
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
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2J
#hidden_dense/BiasAdd/ReadVariableOp#hidden_dense/BiasAdd/ReadVariableOp2H
"hidden_dense/MatMul/ReadVariableOp"hidden_dense/MatMul/ReadVariableOp2V
)hidden_dense_value/BiasAdd/ReadVariableOp)hidden_dense_value/BiasAdd/ReadVariableOp2T
(hidden_dense_value/MatMul/ReadVariableOp(hidden_dense_value/MatMul/ReadVariableOp2J
#value_output/BiasAdd/ReadVariableOp#value_output/BiasAdd/ReadVariableOp2H
"value_output/MatMul/ReadVariableOp"value_output/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?	
?
K__inference_hidden_dense_layer_call_and_return_conditional_losses_330371289

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
)__inference_conv2_layer_call_fn_330371203

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
D__inference_conv2_layer_call_and_return_conditional_losses_3303711952
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
?
?
D__inference_conv2_layer_call_and_return_conditional_losses_330371195

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
?
'__inference_signature_wrapper_330371475
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
$__inference__wrapped_model_3303711612
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
?
?
0__inference_hidden_dense_layer_call_fn_330371659

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
:??????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*T
fORM
K__inference_hidden_dense_layer_call_and_return_conditional_losses_3303712892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
0__inference_value_output_layer_call_fn_330371694

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
CPU2*0,1,2,3J 8*T
fORM
K__inference_value_output_layer_call_and_return_conditional_losses_3303713342
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
?.
?
B__inference_dqn_layer_call_and_return_conditional_losses_330371372
state_input(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2(
$conv2_statefulpartitionedcall_args_1(
$conv2_statefulpartitionedcall_args_2(
$conv3_statefulpartitionedcall_args_1(
$conv3_statefulpartitionedcall_args_2/
+hidden_dense_statefulpartitionedcall_args_1/
+hidden_dense_statefulpartitionedcall_args_25
1hidden_dense_value_statefulpartitionedcall_args_15
1hidden_dense_value_statefulpartitionedcall_args_2/
+value_output_statefulpartitionedcall_args_1/
+value_output_statefulpartitionedcall_args_2
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?$hidden_dense/StatefulPartitionedCall?*hidden_dense_value/StatefulPartitionedCall?$value_output/StatefulPartitionedCall?
#tf_op_layer_Cast_16/PartitionedCallPartitionedCallstate_input*
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
R__inference_tf_op_layer_Cast_16_layer_call_and_return_conditional_losses_3303712332%
#tf_op_layer_Cast_16/PartitionedCall?
&tf_op_layer_truediv_16/PartitionedCallPartitionedCall,tf_op_layer_Cast_16/PartitionedCall:output:0*
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
U__inference_tf_op_layer_truediv_16_layer_call_and_return_conditional_losses_3303712472(
&tf_op_layer_truediv_16/PartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_truediv_16/PartitionedCall:output:0$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*
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
D__inference_conv1_layer_call_and_return_conditional_losses_3303711742
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
D__inference_conv2_layer_call_and_return_conditional_losses_3303711952
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
D__inference_conv3_layer_call_and_return_conditional_losses_3303712162
conv3/StatefulPartitionedCall?
flatten_16/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
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
I__inference_flatten_16_layer_call_and_return_conditional_losses_3303712702
flatten_16/PartitionedCall?
$hidden_dense/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0+hidden_dense_statefulpartitionedcall_args_1+hidden_dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*T
fORM
K__inference_hidden_dense_layer_call_and_return_conditional_losses_3303712892&
$hidden_dense/StatefulPartitionedCall?
*hidden_dense_value/StatefulPartitionedCallStatefulPartitionedCall-hidden_dense/StatefulPartitionedCall:output:01hidden_dense_value_statefulpartitionedcall_args_11hidden_dense_value_statefulpartitionedcall_args_2*
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
CPU2*0,1,2,3J 8*Z
fURS
Q__inference_hidden_dense_value_layer_call_and_return_conditional_losses_3303713122,
*hidden_dense_value/StatefulPartitionedCall?
$value_output/StatefulPartitionedCallStatefulPartitionedCall3hidden_dense_value/StatefulPartitionedCall:output:0+value_output_statefulpartitionedcall_args_1+value_output_statefulpartitionedcall_args_2*
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
CPU2*0,1,2,3J 8*T
fORM
K__inference_value_output_layer_call_and_return_conditional_losses_3303713342&
$value_output/StatefulPartitionedCall?
IdentityIdentity-value_output/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall%^hidden_dense/StatefulPartitionedCall+^hidden_dense_value/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????TT::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2L
$hidden_dense/StatefulPartitionedCall$hidden_dense/StatefulPartitionedCall2X
*hidden_dense_value/StatefulPartitionedCall*hidden_dense_value/StatefulPartitionedCall2L
$value_output/StatefulPartitionedCall$value_output/StatefulPartitionedCall:+ '
%
_user_specified_namestate_input
?
s
U__inference_tf_op_layer_truediv_16_layer_call_and_return_conditional_losses_330371625
inputs_0
identitya
truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
truediv_16/y?

truediv_16RealDivinputs_0truediv_16/y:output:0*
T0*
_cloned(*/
_output_shapes
:?????????TT2

truediv_16j
IdentityIdentitytruediv_16:z:0*
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
n
R__inference_tf_op_layer_Cast_16_layer_call_and_return_conditional_losses_330371233

inputs
identityz
Cast_16Castinputs*

DstT0*

SrcT0*
_cloned(*/
_output_shapes
:?????????TT2	
Cast_16g
IdentityIdentityCast_16:y:0*
T0*/
_output_shapes
:?????????TT2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????TT:& "
 
_user_specified_nameinputs
?
?
D__inference_conv1_layer_call_and_return_conditional_losses_330371174

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
?	
?
Q__inference_hidden_dense_value_layer_call_and_return_conditional_losses_330371670

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
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
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
X
:__inference_tf_op_layer_truediv_16_layer_call_fn_330371630
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
U__inference_tf_op_layer_truediv_16_layer_call_and_return_conditional_losses_3303712472
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
D__inference_conv3_layer_call_and_return_conditional_losses_330371216

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
?
p
R__inference_tf_op_layer_Cast_16_layer_call_and_return_conditional_losses_330371614
inputs_0
identity|
Cast_16Castinputs_0*

DstT0*

SrcT0*
_cloned(*/
_output_shapes
:?????????TT2	
Cast_16g
IdentityIdentityCast_16:y:0*
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
K__inference_hidden_dense_layer_call_and_return_conditional_losses_330371652

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?7
?
%__inference__traced_restore_330371802
file_prefix$
 assignvariableop_conv1_16_kernel$
 assignvariableop_1_conv1_16_bias&
"assignvariableop_2_conv2_16_kernel$
 assignvariableop_3_conv2_16_bias&
"assignvariableop_4_conv3_16_kernel$
 assignvariableop_5_conv3_16_bias-
)assignvariableop_6_hidden_dense_16_kernel+
'assignvariableop_7_hidden_dense_16_bias3
/assignvariableop_8_hidden_dense_value_16_kernel1
-assignvariableop_9_hidden_dense_value_16_bias-
)assignvariableop_10_value_output_6_kernel+
'assignvariableop_11_value_output_6_bias
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
AssignVariableOpAssignVariableOp assignvariableop_conv1_16_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1_16_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2_16_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2_16_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv3_16_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv3_16_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp)assignvariableop_6_hidden_dense_16_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp'assignvariableop_7_hidden_dense_16_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_hidden_dense_value_16_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp-assignvariableop_9_hidden_dense_value_16_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp)assignvariableop_10_value_output_6_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp'assignvariableop_11_value_output_6_biasIdentity_11:output:0*
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
?A
?
B__inference_dqn_layer_call_and_return_conditional_losses_330371575

inputs(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource/
+hidden_dense_matmul_readvariableop_resource0
,hidden_dense_biasadd_readvariableop_resource5
1hidden_dense_value_matmul_readvariableop_resource6
2hidden_dense_value_biasadd_readvariableop_resource/
+value_output_matmul_readvariableop_resource0
,value_output_biasadd_readvariableop_resource
identity??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?conv3/BiasAdd/ReadVariableOp?conv3/Conv2D/ReadVariableOp?#hidden_dense/BiasAdd/ReadVariableOp?"hidden_dense/MatMul/ReadVariableOp?)hidden_dense_value/BiasAdd/ReadVariableOp?(hidden_dense_value/MatMul/ReadVariableOp?#value_output/BiasAdd/ReadVariableOp?"value_output/MatMul/ReadVariableOp?
tf_op_layer_Cast_16/Cast_16Castinputs*

DstT0*

SrcT0*
_cloned(*/
_output_shapes
:?????????TT2
tf_op_layer_Cast_16/Cast_16?
#tf_op_layer_truediv_16/truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2%
#tf_op_layer_truediv_16/truediv_16/y?
!tf_op_layer_truediv_16/truediv_16RealDivtf_op_layer_Cast_16/Cast_16:y:0,tf_op_layer_truediv_16/truediv_16/y:output:0*
T0*
_cloned(*/
_output_shapes
:?????????TT2#
!tf_op_layer_truediv_16/truediv_16?
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2D%tf_op_layer_truediv_16/truediv_16:z:0#conv1/Conv2D/ReadVariableOp:value:0*
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
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_16/Const?
flatten_16/ReshapeReshapeconv3/Relu:activations:0flatten_16/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_16/Reshape?
"hidden_dense/MatMul/ReadVariableOpReadVariableOp+hidden_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"hidden_dense/MatMul/ReadVariableOp?
hidden_dense/MatMulMatMulflatten_16/Reshape:output:0*hidden_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_dense/MatMul?
#hidden_dense/BiasAdd/ReadVariableOpReadVariableOp,hidden_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#hidden_dense/BiasAdd/ReadVariableOp?
hidden_dense/BiasAddBiasAddhidden_dense/MatMul:product:0+hidden_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_dense/BiasAdd?
hidden_dense/TanhTanhhidden_dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
hidden_dense/Tanh?
(hidden_dense_value/MatMul/ReadVariableOpReadVariableOp1hidden_dense_value_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(hidden_dense_value/MatMul/ReadVariableOp?
hidden_dense_value/MatMulMatMulhidden_dense/Tanh:y:00hidden_dense_value/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_dense_value/MatMul?
)hidden_dense_value/BiasAdd/ReadVariableOpReadVariableOp2hidden_dense_value_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)hidden_dense_value/BiasAdd/ReadVariableOp?
hidden_dense_value/BiasAddBiasAdd#hidden_dense_value/MatMul:product:01hidden_dense_value/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
hidden_dense_value/BiasAdd?
hidden_dense_value/TanhTanh#hidden_dense_value/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
hidden_dense_value/Tanh?
"value_output/MatMul/ReadVariableOpReadVariableOp+value_output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"value_output/MatMul/ReadVariableOp?
value_output/MatMulMatMulhidden_dense_value/Tanh:y:0*value_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
value_output/MatMul?
#value_output/BiasAdd/ReadVariableOpReadVariableOp,value_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#value_output/BiasAdd/ReadVariableOp?
value_output/BiasAddBiasAddvalue_output/MatMul:product:0+value_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
value_output/BiasAdd?
IdentityIdentityvalue_output/BiasAdd:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp$^hidden_dense/BiasAdd/ReadVariableOp#^hidden_dense/MatMul/ReadVariableOp*^hidden_dense_value/BiasAdd/ReadVariableOp)^hidden_dense_value/MatMul/ReadVariableOp$^value_output/BiasAdd/ReadVariableOp#^value_output/MatMul/ReadVariableOp*
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
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2J
#hidden_dense/BiasAdd/ReadVariableOp#hidden_dense/BiasAdd/ReadVariableOp2H
"hidden_dense/MatMul/ReadVariableOp"hidden_dense/MatMul/ReadVariableOp2V
)hidden_dense_value/BiasAdd/ReadVariableOp)hidden_dense_value/BiasAdd/ReadVariableOp2T
(hidden_dense_value/MatMul/ReadVariableOp(hidden_dense_value/MatMul/ReadVariableOp2J
#value_output/BiasAdd/ReadVariableOp#value_output/BiasAdd/ReadVariableOp2H
"value_output/MatMul/ReadVariableOp"value_output/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?G
?	
$__inference__wrapped_model_330371161
state_input,
(dqn_conv1_conv2d_readvariableop_resource-
)dqn_conv1_biasadd_readvariableop_resource,
(dqn_conv2_conv2d_readvariableop_resource-
)dqn_conv2_biasadd_readvariableop_resource,
(dqn_conv3_conv2d_readvariableop_resource-
)dqn_conv3_biasadd_readvariableop_resource3
/dqn_hidden_dense_matmul_readvariableop_resource4
0dqn_hidden_dense_biasadd_readvariableop_resource9
5dqn_hidden_dense_value_matmul_readvariableop_resource:
6dqn_hidden_dense_value_biasadd_readvariableop_resource3
/dqn_value_output_matmul_readvariableop_resource4
0dqn_value_output_biasadd_readvariableop_resource
identity?? dqn/conv1/BiasAdd/ReadVariableOp?dqn/conv1/Conv2D/ReadVariableOp? dqn/conv2/BiasAdd/ReadVariableOp?dqn/conv2/Conv2D/ReadVariableOp? dqn/conv3/BiasAdd/ReadVariableOp?dqn/conv3/Conv2D/ReadVariableOp?'dqn/hidden_dense/BiasAdd/ReadVariableOp?&dqn/hidden_dense/MatMul/ReadVariableOp?-dqn/hidden_dense_value/BiasAdd/ReadVariableOp?,dqn/hidden_dense_value/MatMul/ReadVariableOp?'dqn/value_output/BiasAdd/ReadVariableOp?&dqn/value_output/MatMul/ReadVariableOp?
dqn/tf_op_layer_Cast_16/Cast_16Caststate_input*

DstT0*

SrcT0*
_cloned(*/
_output_shapes
:?????????TT2!
dqn/tf_op_layer_Cast_16/Cast_16?
'dqn/tf_op_layer_truediv_16/truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2)
'dqn/tf_op_layer_truediv_16/truediv_16/y?
%dqn/tf_op_layer_truediv_16/truediv_16RealDiv#dqn/tf_op_layer_Cast_16/Cast_16:y:00dqn/tf_op_layer_truediv_16/truediv_16/y:output:0*
T0*
_cloned(*/
_output_shapes
:?????????TT2'
%dqn/tf_op_layer_truediv_16/truediv_16?
dqn/conv1/Conv2D/ReadVariableOpReadVariableOp(dqn_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
dqn/conv1/Conv2D/ReadVariableOp?
dqn/conv1/Conv2DConv2D)dqn/tf_op_layer_truediv_16/truediv_16:z:0'dqn/conv1/Conv2D/ReadVariableOp:value:0*
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
dqn/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
dqn/flatten_16/Const?
dqn/flatten_16/ReshapeReshapedqn/conv3/Relu:activations:0dqn/flatten_16/Const:output:0*
T0*(
_output_shapes
:??????????2
dqn/flatten_16/Reshape?
&dqn/hidden_dense/MatMul/ReadVariableOpReadVariableOp/dqn_hidden_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&dqn/hidden_dense/MatMul/ReadVariableOp?
dqn/hidden_dense/MatMulMatMuldqn/flatten_16/Reshape:output:0.dqn/hidden_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dqn/hidden_dense/MatMul?
'dqn/hidden_dense/BiasAdd/ReadVariableOpReadVariableOp0dqn_hidden_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'dqn/hidden_dense/BiasAdd/ReadVariableOp?
dqn/hidden_dense/BiasAddBiasAdd!dqn/hidden_dense/MatMul:product:0/dqn/hidden_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dqn/hidden_dense/BiasAdd?
dqn/hidden_dense/TanhTanh!dqn/hidden_dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dqn/hidden_dense/Tanh?
,dqn/hidden_dense_value/MatMul/ReadVariableOpReadVariableOp5dqn_hidden_dense_value_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,dqn/hidden_dense_value/MatMul/ReadVariableOp?
dqn/hidden_dense_value/MatMulMatMuldqn/hidden_dense/Tanh:y:04dqn/hidden_dense_value/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dqn/hidden_dense_value/MatMul?
-dqn/hidden_dense_value/BiasAdd/ReadVariableOpReadVariableOp6dqn_hidden_dense_value_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-dqn/hidden_dense_value/BiasAdd/ReadVariableOp?
dqn/hidden_dense_value/BiasAddBiasAdd'dqn/hidden_dense_value/MatMul:product:05dqn/hidden_dense_value/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
dqn/hidden_dense_value/BiasAdd?
dqn/hidden_dense_value/TanhTanh'dqn/hidden_dense_value/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dqn/hidden_dense_value/Tanh?
&dqn/value_output/MatMul/ReadVariableOpReadVariableOp/dqn_value_output_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&dqn/value_output/MatMul/ReadVariableOp?
dqn/value_output/MatMulMatMuldqn/hidden_dense_value/Tanh:y:0.dqn/value_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dqn/value_output/MatMul?
'dqn/value_output/BiasAdd/ReadVariableOpReadVariableOp0dqn_value_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'dqn/value_output/BiasAdd/ReadVariableOp?
dqn/value_output/BiasAddBiasAdd!dqn/value_output/MatMul:product:0/dqn/value_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dqn/value_output/BiasAdd?
IdentityIdentity!dqn/value_output/BiasAdd:output:0!^dqn/conv1/BiasAdd/ReadVariableOp ^dqn/conv1/Conv2D/ReadVariableOp!^dqn/conv2/BiasAdd/ReadVariableOp ^dqn/conv2/Conv2D/ReadVariableOp!^dqn/conv3/BiasAdd/ReadVariableOp ^dqn/conv3/Conv2D/ReadVariableOp(^dqn/hidden_dense/BiasAdd/ReadVariableOp'^dqn/hidden_dense/MatMul/ReadVariableOp.^dqn/hidden_dense_value/BiasAdd/ReadVariableOp-^dqn/hidden_dense_value/MatMul/ReadVariableOp(^dqn/value_output/BiasAdd/ReadVariableOp'^dqn/value_output/MatMul/ReadVariableOp*
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
dqn/conv3/Conv2D/ReadVariableOpdqn/conv3/Conv2D/ReadVariableOp2R
'dqn/hidden_dense/BiasAdd/ReadVariableOp'dqn/hidden_dense/BiasAdd/ReadVariableOp2P
&dqn/hidden_dense/MatMul/ReadVariableOp&dqn/hidden_dense/MatMul/ReadVariableOp2^
-dqn/hidden_dense_value/BiasAdd/ReadVariableOp-dqn/hidden_dense_value/BiasAdd/ReadVariableOp2\
,dqn/hidden_dense_value/MatMul/ReadVariableOp,dqn/hidden_dense_value/MatMul/ReadVariableOp2R
'dqn/value_output/BiasAdd/ReadVariableOp'dqn/value_output/BiasAdd/ReadVariableOp2P
&dqn/value_output/MatMul/ReadVariableOp&dqn/value_output/MatMul/ReadVariableOp:+ '
%
_user_specified_namestate_input
?
U
7__inference_tf_op_layer_Cast_16_layer_call_fn_330371619
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
R__inference_tf_op_layer_Cast_16_layer_call_and_return_conditional_losses_3303712332
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
?
q
U__inference_tf_op_layer_truediv_16_layer_call_and_return_conditional_losses_330371247

inputs
identitya
truediv_16/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
truediv_16/y?

truediv_16RealDivinputstruediv_16/y:output:0*
T0*
_cloned(*/
_output_shapes
:?????????TT2

truediv_16j
IdentityIdentitytruediv_16:z:0*
T0*/
_output_shapes
:?????????TT2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????TT:& "
 
_user_specified_nameinputs
?.
?
B__inference_dqn_layer_call_and_return_conditional_losses_330371442

inputs(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2(
$conv2_statefulpartitionedcall_args_1(
$conv2_statefulpartitionedcall_args_2(
$conv3_statefulpartitionedcall_args_1(
$conv3_statefulpartitionedcall_args_2/
+hidden_dense_statefulpartitionedcall_args_1/
+hidden_dense_statefulpartitionedcall_args_25
1hidden_dense_value_statefulpartitionedcall_args_15
1hidden_dense_value_statefulpartitionedcall_args_2/
+value_output_statefulpartitionedcall_args_1/
+value_output_statefulpartitionedcall_args_2
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?$hidden_dense/StatefulPartitionedCall?*hidden_dense_value/StatefulPartitionedCall?$value_output/StatefulPartitionedCall?
#tf_op_layer_Cast_16/PartitionedCallPartitionedCallinputs*
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
R__inference_tf_op_layer_Cast_16_layer_call_and_return_conditional_losses_3303712332%
#tf_op_layer_Cast_16/PartitionedCall?
&tf_op_layer_truediv_16/PartitionedCallPartitionedCall,tf_op_layer_Cast_16/PartitionedCall:output:0*
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
U__inference_tf_op_layer_truediv_16_layer_call_and_return_conditional_losses_3303712472(
&tf_op_layer_truediv_16/PartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_truediv_16/PartitionedCall:output:0$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*
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
D__inference_conv1_layer_call_and_return_conditional_losses_3303711742
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
D__inference_conv2_layer_call_and_return_conditional_losses_3303711952
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
D__inference_conv3_layer_call_and_return_conditional_losses_3303712162
conv3/StatefulPartitionedCall?
flatten_16/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
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
I__inference_flatten_16_layer_call_and_return_conditional_losses_3303712702
flatten_16/PartitionedCall?
$hidden_dense/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0+hidden_dense_statefulpartitionedcall_args_1+hidden_dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*T
fORM
K__inference_hidden_dense_layer_call_and_return_conditional_losses_3303712892&
$hidden_dense/StatefulPartitionedCall?
*hidden_dense_value/StatefulPartitionedCallStatefulPartitionedCall-hidden_dense/StatefulPartitionedCall:output:01hidden_dense_value_statefulpartitionedcall_args_11hidden_dense_value_statefulpartitionedcall_args_2*
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
CPU2*0,1,2,3J 8*Z
fURS
Q__inference_hidden_dense_value_layer_call_and_return_conditional_losses_3303713122,
*hidden_dense_value/StatefulPartitionedCall?
$value_output/StatefulPartitionedCallStatefulPartitionedCall3hidden_dense_value/StatefulPartitionedCall:output:0+value_output_statefulpartitionedcall_args_1+value_output_statefulpartitionedcall_args_2*
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
CPU2*0,1,2,3J 8*T
fORM
K__inference_value_output_layer_call_and_return_conditional_losses_3303713342&
$value_output/StatefulPartitionedCall?
IdentityIdentity-value_output/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall%^hidden_dense/StatefulPartitionedCall+^hidden_dense_value/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????TT::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2L
$hidden_dense/StatefulPartitionedCall$hidden_dense/StatefulPartitionedCall2X
*hidden_dense_value/StatefulPartitionedCall*hidden_dense_value/StatefulPartitionedCall2L
$value_output/StatefulPartitionedCall$value_output/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
)__inference_conv1_layer_call_fn_330371182

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
D__inference_conv1_layer_call_and_return_conditional_losses_3303711742
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
 
_user_specified_nameinputs
?
?
'__inference_dqn_layer_call_fn_330371457
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
B__inference_dqn_layer_call_and_return_conditional_losses_3303714422
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
?.
?
B__inference_dqn_layer_call_and_return_conditional_losses_330371400

inputs(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2(
$conv2_statefulpartitionedcall_args_1(
$conv2_statefulpartitionedcall_args_2(
$conv3_statefulpartitionedcall_args_1(
$conv3_statefulpartitionedcall_args_2/
+hidden_dense_statefulpartitionedcall_args_1/
+hidden_dense_statefulpartitionedcall_args_25
1hidden_dense_value_statefulpartitionedcall_args_15
1hidden_dense_value_statefulpartitionedcall_args_2/
+value_output_statefulpartitionedcall_args_1/
+value_output_statefulpartitionedcall_args_2
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?$hidden_dense/StatefulPartitionedCall?*hidden_dense_value/StatefulPartitionedCall?$value_output/StatefulPartitionedCall?
#tf_op_layer_Cast_16/PartitionedCallPartitionedCallinputs*
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
R__inference_tf_op_layer_Cast_16_layer_call_and_return_conditional_losses_3303712332%
#tf_op_layer_Cast_16/PartitionedCall?
&tf_op_layer_truediv_16/PartitionedCallPartitionedCall,tf_op_layer_Cast_16/PartitionedCall:output:0*
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
U__inference_tf_op_layer_truediv_16_layer_call_and_return_conditional_losses_3303712472(
&tf_op_layer_truediv_16/PartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_truediv_16/PartitionedCall:output:0$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*
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
D__inference_conv1_layer_call_and_return_conditional_losses_3303711742
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
D__inference_conv2_layer_call_and_return_conditional_losses_3303711952
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
D__inference_conv3_layer_call_and_return_conditional_losses_3303712162
conv3/StatefulPartitionedCall?
flatten_16/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
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
I__inference_flatten_16_layer_call_and_return_conditional_losses_3303712702
flatten_16/PartitionedCall?
$hidden_dense/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0+hidden_dense_statefulpartitionedcall_args_1+hidden_dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*T
fORM
K__inference_hidden_dense_layer_call_and_return_conditional_losses_3303712892&
$hidden_dense/StatefulPartitionedCall?
*hidden_dense_value/StatefulPartitionedCallStatefulPartitionedCall-hidden_dense/StatefulPartitionedCall:output:01hidden_dense_value_statefulpartitionedcall_args_11hidden_dense_value_statefulpartitionedcall_args_2*
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
CPU2*0,1,2,3J 8*Z
fURS
Q__inference_hidden_dense_value_layer_call_and_return_conditional_losses_3303713122,
*hidden_dense_value/StatefulPartitionedCall?
$value_output/StatefulPartitionedCallStatefulPartitionedCall3hidden_dense_value/StatefulPartitionedCall:output:0+value_output_statefulpartitionedcall_args_1+value_output_statefulpartitionedcall_args_2*
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
CPU2*0,1,2,3J 8*T
fORM
K__inference_value_output_layer_call_and_return_conditional_losses_3303713342&
$value_output/StatefulPartitionedCall?
IdentityIdentity-value_output/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall%^hidden_dense/StatefulPartitionedCall+^hidden_dense_value/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????TT::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2L
$hidden_dense/StatefulPartitionedCall$hidden_dense/StatefulPartitionedCall2X
*hidden_dense_value/StatefulPartitionedCall*hidden_dense_value/StatefulPartitionedCall2L
$value_output/StatefulPartitionedCall$value_output/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
6__inference_hidden_dense_value_layer_call_fn_330371677

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
CPU2*0,1,2,3J 8*Z
fURS
Q__inference_hidden_dense_value_layer_call_and_return_conditional_losses_3303713122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
e
I__inference_flatten_16_layer_call_and_return_conditional_losses_330371270

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
?
e
I__inference_flatten_16_layer_call_and_return_conditional_losses_330371636

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
?.
?
B__inference_dqn_layer_call_and_return_conditional_losses_330371347
state_input(
$conv1_statefulpartitionedcall_args_1(
$conv1_statefulpartitionedcall_args_2(
$conv2_statefulpartitionedcall_args_1(
$conv2_statefulpartitionedcall_args_2(
$conv3_statefulpartitionedcall_args_1(
$conv3_statefulpartitionedcall_args_2/
+hidden_dense_statefulpartitionedcall_args_1/
+hidden_dense_statefulpartitionedcall_args_25
1hidden_dense_value_statefulpartitionedcall_args_15
1hidden_dense_value_statefulpartitionedcall_args_2/
+value_output_statefulpartitionedcall_args_1/
+value_output_statefulpartitionedcall_args_2
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?conv3/StatefulPartitionedCall?$hidden_dense/StatefulPartitionedCall?*hidden_dense_value/StatefulPartitionedCall?$value_output/StatefulPartitionedCall?
#tf_op_layer_Cast_16/PartitionedCallPartitionedCallstate_input*
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
R__inference_tf_op_layer_Cast_16_layer_call_and_return_conditional_losses_3303712332%
#tf_op_layer_Cast_16/PartitionedCall?
&tf_op_layer_truediv_16/PartitionedCallPartitionedCall,tf_op_layer_Cast_16/PartitionedCall:output:0*
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
U__inference_tf_op_layer_truediv_16_layer_call_and_return_conditional_losses_3303712472(
&tf_op_layer_truediv_16/PartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_truediv_16/PartitionedCall:output:0$conv1_statefulpartitionedcall_args_1$conv1_statefulpartitionedcall_args_2*
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
D__inference_conv1_layer_call_and_return_conditional_losses_3303711742
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
D__inference_conv2_layer_call_and_return_conditional_losses_3303711952
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
D__inference_conv3_layer_call_and_return_conditional_losses_3303712162
conv3/StatefulPartitionedCall?
flatten_16/PartitionedCallPartitionedCall&conv3/StatefulPartitionedCall:output:0*
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
I__inference_flatten_16_layer_call_and_return_conditional_losses_3303712702
flatten_16/PartitionedCall?
$hidden_dense/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0+hidden_dense_statefulpartitionedcall_args_1+hidden_dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????*3
config_proto#!

GPU

CPU2*0,1,2,3J 8*T
fORM
K__inference_hidden_dense_layer_call_and_return_conditional_losses_3303712892&
$hidden_dense/StatefulPartitionedCall?
*hidden_dense_value/StatefulPartitionedCallStatefulPartitionedCall-hidden_dense/StatefulPartitionedCall:output:01hidden_dense_value_statefulpartitionedcall_args_11hidden_dense_value_statefulpartitionedcall_args_2*
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
CPU2*0,1,2,3J 8*Z
fURS
Q__inference_hidden_dense_value_layer_call_and_return_conditional_losses_3303713122,
*hidden_dense_value/StatefulPartitionedCall?
$value_output/StatefulPartitionedCallStatefulPartitionedCall3hidden_dense_value/StatefulPartitionedCall:output:0+value_output_statefulpartitionedcall_args_1+value_output_statefulpartitionedcall_args_2*
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
CPU2*0,1,2,3J 8*T
fORM
K__inference_value_output_layer_call_and_return_conditional_losses_3303713342&
$value_output/StatefulPartitionedCall?
IdentityIdentity-value_output/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^conv3/StatefulPartitionedCall%^hidden_dense/StatefulPartitionedCall+^hidden_dense_value/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????TT::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2L
$hidden_dense/StatefulPartitionedCall$hidden_dense/StatefulPartitionedCall2X
*hidden_dense_value/StatefulPartitionedCall*hidden_dense_value/StatefulPartitionedCall2L
$value_output/StatefulPartitionedCall$value_output/StatefulPartitionedCall:+ '
%
_user_specified_namestate_input
?&
?
"__inference__traced_save_330371754
file_prefix.
*savev2_conv1_16_kernel_read_readvariableop,
(savev2_conv1_16_bias_read_readvariableop.
*savev2_conv2_16_kernel_read_readvariableop,
(savev2_conv2_16_bias_read_readvariableop.
*savev2_conv3_16_kernel_read_readvariableop,
(savev2_conv3_16_bias_read_readvariableop5
1savev2_hidden_dense_16_kernel_read_readvariableop3
/savev2_hidden_dense_16_bias_read_readvariableop;
7savev2_hidden_dense_value_16_kernel_read_readvariableop9
5savev2_hidden_dense_value_16_bias_read_readvariableop4
0savev2_value_output_6_kernel_read_readvariableop2
.savev2_value_output_6_bias_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_e5288d8145bb4b0cb3b3df780923be05/part2
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1_16_kernel_read_readvariableop(savev2_conv1_16_bias_read_readvariableop*savev2_conv2_16_kernel_read_readvariableop(savev2_conv2_16_bias_read_readvariableop*savev2_conv3_16_kernel_read_readvariableop(savev2_conv3_16_bias_read_readvariableop1savev2_hidden_dense_16_kernel_read_readvariableop/savev2_hidden_dense_16_bias_read_readvariableop7savev2_hidden_dense_value_16_kernel_read_readvariableop5savev2_hidden_dense_value_16_bias_read_readvariableop0savev2_value_output_6_kernel_read_readvariableop.savev2_value_output_6_bias_read_readvariableop"/device:CPU:0*
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
??:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
?
?
K__inference_value_output_layer_call_and_return_conditional_losses_330371687

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
serving_default_state_input:0?????????TT@
value_output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?M
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

layer_with_weights-5

layer-9
	variables
regularization_losses
trainable_variables
	keras_api

signatures
j__call__
*k&call_and_return_all_conditional_losses
l_default_save_signature"?J
_tf_keras_model?I{"class_name": "Model", "name": "dqn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dqn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 84, 84, 4], "dtype": "uint8", "sparse": false, "ragged": false, "name": "state_input"}, "name": "state_input", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Cast_16", "trainable": true, "dtype": "float32", "node_def": {"name": "Cast_16", "op": "Cast", "input": ["state_input_16"], "attr": {"Truncate": {"b": false}, "DstT": {"type": "DT_FLOAT"}, "SrcT": {"type": "DT_UINT8"}}}, "constants": {}}, "name": "tf_op_layer_Cast_16", "inbound_nodes": [[["state_input", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "truediv_16", "trainable": true, "dtype": "float32", "node_def": {"name": "truediv_16", "op": "RealDiv", "input": ["Cast_16", "truediv_16/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 255.0}}, "name": "tf_op_layer_truediv_16", "inbound_nodes": [[["tf_op_layer_Cast_16", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [8, 8], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["tf_op_layer_truediv_16", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [4, 4], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_16", "inbound_nodes": [[["conv3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_dense", "inbound_nodes": [[["flatten_16", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_dense_value", "trainable": true, "dtype": "float32", "units": 512, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_dense_value", "inbound_nodes": [[["hidden_dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "value_output", "trainable": true, "dtype": "float32", "units": 18, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "value_output", "inbound_nodes": [[["hidden_dense_value", 0, 0, {}]]]}], "input_layers": [["state_input", 0, 0]], "output_layers": [["value_output", 0, 0]]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "dqn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 84, 84, 4], "dtype": "uint8", "sparse": false, "ragged": false, "name": "state_input"}, "name": "state_input", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Cast_16", "trainable": true, "dtype": "float32", "node_def": {"name": "Cast_16", "op": "Cast", "input": ["state_input_16"], "attr": {"Truncate": {"b": false}, "DstT": {"type": "DT_FLOAT"}, "SrcT": {"type": "DT_UINT8"}}}, "constants": {}}, "name": "tf_op_layer_Cast_16", "inbound_nodes": [[["state_input", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "truediv_16", "trainable": true, "dtype": "float32", "node_def": {"name": "truediv_16", "op": "RealDiv", "input": ["Cast_16", "truediv_16/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 255.0}}, "name": "tf_op_layer_truediv_16", "inbound_nodes": [[["tf_op_layer_Cast_16", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [8, 8], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["tf_op_layer_truediv_16", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [4, 4], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_16", "inbound_nodes": [[["conv3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_dense", "inbound_nodes": [[["flatten_16", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "hidden_dense_value", "trainable": true, "dtype": "float32", "units": 512, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hidden_dense_value", "inbound_nodes": [[["hidden_dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "value_output", "trainable": true, "dtype": "float32", "units": 18, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "value_output", "inbound_nodes": [[["hidden_dense_value", 0, 0, {}]]]}], "input_layers": [["state_input", 0, 0]], "output_layers": [["value_output", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "state_input", "dtype": "uint8", "sparse": false, "ragged": false, "batch_input_shape": [null, 84, 84, 4], "config": {"batch_input_shape": [null, 84, 84, 4], "dtype": "uint8", "sparse": false, "ragged": false, "name": "state_input"}}
?
	constants
	variables
regularization_losses
trainable_variables
	keras_api
m__call__
*n&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Cast_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Cast_16", "trainable": true, "dtype": "float32", "node_def": {"name": "Cast_16", "op": "Cast", "input": ["state_input_16"], "attr": {"Truncate": {"b": false}, "DstT": {"type": "DT_FLOAT"}, "SrcT": {"type": "DT_UINT8"}}}, "constants": {}}}
?
	constants
	variables
regularization_losses
trainable_variables
	keras_api
o__call__
*p&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_truediv_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "truediv_16", "trainable": true, "dtype": "float32", "node_def": {"name": "truediv_16", "op": "RealDiv", "input": ["Cast_16", "truediv_16/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 255.0}}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
q__call__
*r&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [8, 8], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}}}
?

 kernel
!bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
s__call__
*t&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [4, 4], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}}
?

&kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
u__call__
*v&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
?
,	variables
-regularization_losses
.trainable_variables
/	keras_api
w__call__
*x&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
y__call__
*z&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden_dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1568}}}}
?

6kernel
7bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api
{__call__
*|&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden_dense_value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_dense_value", "trainable": true, "dtype": "float32", "units": 512, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
?

<kernel
=bias
>	variables
?regularization_losses
@trainable_variables
A	keras_api
}__call__
*~&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "value_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "value_output", "trainable": true, "dtype": "float32", "units": 18, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}}
v
0
1
 2
!3
&4
'5
06
17
68
79
<10
=11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
 2
!3
&4
'5
06
17
68
79
<10
=11"
trackable_list_wrapper
?
	variables
regularization_losses
Bmetrics
Cnon_trainable_variables

Dlayers
Elayer_regularization_losses
trainable_variables
j__call__
l_default_save_signature
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
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
	variables
regularization_losses
Fmetrics
Gnon_trainable_variables
Hlayer_regularization_losses

Ilayers
trainable_variables
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
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
	variables
regularization_losses
Jmetrics
Knon_trainable_variables
Llayer_regularization_losses

Mlayers
trainable_variables
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
):'2conv1_16/kernel
:2conv1_16/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
regularization_losses
Nmetrics
Onon_trainable_variables
Player_regularization_losses

Qlayers
trainable_variables
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2_16/kernel
: 2conv2_16/bias
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
?
"	variables
#regularization_losses
Rmetrics
Snon_trainable_variables
Tlayer_regularization_losses

Ulayers
$trainable_variables
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
):'  2conv3_16/kernel
: 2conv3_16/bias
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
?
(	variables
)regularization_losses
Vmetrics
Wnon_trainable_variables
Xlayer_regularization_losses

Ylayers
*trainable_variables
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
,	variables
-regularization_losses
Zmetrics
[non_trainable_variables
\layer_regularization_losses

]layers
.trainable_variables
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
*:(
??2hidden_dense_16/kernel
#:!?2hidden_dense_16/bias
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?
2	variables
3regularization_losses
^metrics
_non_trainable_variables
`layer_regularization_losses

alayers
4trainable_variables
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
0:.
??2hidden_dense_value_16/kernel
):'?2hidden_dense_value_16/bias
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
?
8	variables
9regularization_losses
bmetrics
cnon_trainable_variables
dlayer_regularization_losses

elayers
:trainable_variables
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
(:&	?2value_output_6/kernel
!:2value_output_6/bias
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
?
>	variables
?regularization_losses
fmetrics
gnon_trainable_variables
hlayer_regularization_losses

ilayers
@trainable_variables
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
f
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
9"
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
'__inference_dqn_layer_call_fn_330371592
'__inference_dqn_layer_call_fn_330371457
'__inference_dqn_layer_call_fn_330371415
'__inference_dqn_layer_call_fn_330371609?
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
B__inference_dqn_layer_call_and_return_conditional_losses_330371575
B__inference_dqn_layer_call_and_return_conditional_losses_330371372
B__inference_dqn_layer_call_and_return_conditional_losses_330371347
B__inference_dqn_layer_call_and_return_conditional_losses_330371525?
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
$__inference__wrapped_model_330371161?
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
7__inference_tf_op_layer_Cast_16_layer_call_fn_330371619?
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
R__inference_tf_op_layer_Cast_16_layer_call_and_return_conditional_losses_330371614?
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
:__inference_tf_op_layer_truediv_16_layer_call_fn_330371630?
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
U__inference_tf_op_layer_truediv_16_layer_call_and_return_conditional_losses_330371625?
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
)__inference_conv1_layer_call_fn_330371182?
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
D__inference_conv1_layer_call_and_return_conditional_losses_330371174?
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
)__inference_conv2_layer_call_fn_330371203?
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
D__inference_conv2_layer_call_and_return_conditional_losses_330371195?
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
)__inference_conv3_layer_call_fn_330371224?
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
D__inference_conv3_layer_call_and_return_conditional_losses_330371216?
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
.__inference_flatten_16_layer_call_fn_330371641?
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
I__inference_flatten_16_layer_call_and_return_conditional_losses_330371636?
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
0__inference_hidden_dense_layer_call_fn_330371659?
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
K__inference_hidden_dense_layer_call_and_return_conditional_losses_330371652?
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
6__inference_hidden_dense_value_layer_call_fn_330371677?
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
Q__inference_hidden_dense_value_layer_call_and_return_conditional_losses_330371670?
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
0__inference_value_output_layer_call_fn_330371694?
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
K__inference_value_output_layer_call_and_return_conditional_losses_330371687?
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
'__inference_signature_wrapper_330371475state_input?
$__inference__wrapped_model_330371161? !&'0167<=<?9
2?/
-?*
state_input?????????TT
? ";?8
6
value_output&?#
value_output??????????
D__inference_conv1_layer_call_and_return_conditional_losses_330371174?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
)__inference_conv1_layer_call_fn_330371182?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
D__inference_conv2_layer_call_and_return_conditional_losses_330371195? !I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+??????????????????????????? 
? ?
)__inference_conv2_layer_call_fn_330371203? !I?F
??<
:?7
inputs+???????????????????????????
? "2?/+??????????????????????????? ?
D__inference_conv3_layer_call_and_return_conditional_losses_330371216?&'I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
)__inference_conv3_layer_call_fn_330371224?&'I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
B__inference_dqn_layer_call_and_return_conditional_losses_330371347{ !&'0167<=D?A
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
B__inference_dqn_layer_call_and_return_conditional_losses_330371372{ !&'0167<=D?A
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
B__inference_dqn_layer_call_and_return_conditional_losses_330371525v !&'0167<=??<
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
B__inference_dqn_layer_call_and_return_conditional_losses_330371575v !&'0167<=??<
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
'__inference_dqn_layer_call_fn_330371415n !&'0167<=D?A
:?7
-?*
state_input?????????TT
p

 
? "???????????
'__inference_dqn_layer_call_fn_330371457n !&'0167<=D?A
:?7
-?*
state_input?????????TT
p 

 
? "???????????
'__inference_dqn_layer_call_fn_330371592i !&'0167<=??<
5?2
(?%
inputs?????????TT
p

 
? "???????????
'__inference_dqn_layer_call_fn_330371609i !&'0167<=??<
5?2
(?%
inputs?????????TT
p 

 
? "???????????
I__inference_flatten_16_layer_call_and_return_conditional_losses_330371636a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????
? ?
.__inference_flatten_16_layer_call_fn_330371641T7?4
-?*
(?%
inputs????????? 
? "????????????
K__inference_hidden_dense_layer_call_and_return_conditional_losses_330371652^010?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
0__inference_hidden_dense_layer_call_fn_330371659Q010?-
&?#
!?
inputs??????????
? "????????????
Q__inference_hidden_dense_value_layer_call_and_return_conditional_losses_330371670^670?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
6__inference_hidden_dense_value_layer_call_fn_330371677Q670?-
&?#
!?
inputs??????????
? "????????????
'__inference_signature_wrapper_330371475? !&'0167<=K?H
? 
A?>
<
state_input-?*
state_input?????????TT";?8
6
value_output&?#
value_output??????????
R__inference_tf_op_layer_Cast_16_layer_call_and_return_conditional_losses_330371614o>?;
4?1
/?,
*?'
inputs/0?????????TT
? "-?*
#? 
0?????????TT
? ?
7__inference_tf_op_layer_Cast_16_layer_call_fn_330371619b>?;
4?1
/?,
*?'
inputs/0?????????TT
? " ??????????TT?
U__inference_tf_op_layer_truediv_16_layer_call_and_return_conditional_losses_330371625o>?;
4?1
/?,
*?'
inputs/0?????????TT
? "-?*
#? 
0?????????TT
? ?
:__inference_tf_op_layer_truediv_16_layer_call_fn_330371630b>?;
4?1
/?,
*?'
inputs/0?????????TT
? " ??????????TT?
K__inference_value_output_layer_call_and_return_conditional_losses_330371687]<=0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
0__inference_value_output_layer_call_fn_330371694P<=0?-
&?#
!?
inputs??????????
? "??????????