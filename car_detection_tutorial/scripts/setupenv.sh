# Create variables for all models used by the tutorials to make 
#  it easier to reference them with short names

# check for variable set by setupvars.sh in the SDK, need it to find models
: ${InferenceEngine_DIR:?Must source the setupvars.sh in the SDK to set InferenceEngine_DIR}

modelDir=$InferenceEngine_DIR/../../intel_models

# Vehicle and License Plates Detection Model
modName=vehicle-license-plate-detection-barrier-0007
export mVLP16=$modelDir/$modName/FP16/$modName.xml
export mVLP32=$modelDir/$modName/FP32/$modName.xml

# Vehicle Attributes Detection Model
modName=vehicle-attributes-recognition-barrier-0010
export mVA16=$modelDir/$modName/FP16/$modName.xml
export mVA32=$modelDir/$modName/FP32/$modName.xml

# Batch size models (Vehicle Detection, all FP32)
scriptDir=$(dirname "$(readlink -f ${BASH_SOURCE[0]})")
batchModelsDir=$scriptDir/../models/batch_sizes
modName=SSD_GoogleNetV2
export mVB1=$batchModelsDir/batch_1/$modName.xml
export mVB2=$batchModelsDir/batch_2/$modName.xml
export mVB4=$batchModelsDir/batch_4/$modName.xml
export mVB8=$batchModelsDir/batch_8/$modName.xml
export mVB16=$batchModelsDir/batch_16/$modName.xml

