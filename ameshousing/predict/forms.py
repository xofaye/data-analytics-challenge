from django import forms


class PredictPriceForm(forms.Form):
    ms_subclass = forms.CharField(initial="", required=True)
    lot_area = forms.IntegerField(required=True)
    overall_qual = forms.IntegerField(required=True)
    year_built = forms.ChoiceField(choices=[(x, x) for x in range(1980, 2010)])


""" 
MSSubClass
LotFrontage
LotArea
Street
Alley
LotShape
LandSlope
OverallQual
OverallCond
YearBuilt
YearRemodAdd
MasVnrArea
ExterQual
ExterCond
BsmtQual
BsmtCond
BsmtExposure
BsmtFinType1
BsmtFinType2
BsmtUnfSF
TotalBsmtSF
HeatingQC
CentralAir
1stFlrSF
2ndFlrSF
LowQualFinSF
GrLivArea
BsmtFullBath
BsmtHalfBath
FullBath
HalfBath
BedroomAbvGr
KitchenAbvGr
KitchenQual
TotRmsAbvGrd
Functional
Fireplaces
FireplaceQu
GarageYrBlt
GarageFinish
GarageCars
GarageArea
GarageQual
GarageCond
PavedDrive
WoodDeckSF
OpenPorchSF
EnclosedPorch
3SsnPorch
ScreenPorch
PoolArea
PoolQC
Fence
MiscVal
MoSold
YrSold
TotalSF
MSZoning_C (all)
MSZoning_FV
MSZoning_RH
MSZoning_RL
MSZoning_RM
LandContour_Bnk
LandContour_HLS
LandContour_Low
LandContour_Lvl
LotConfig_Corner
LotConfig_CulDSac
LotConfig_FR2
LotConfig_FR3
LotConfig_Inside
Neighborhood_Blmngtn
Neighborhood_Blueste
Neighborhood_BrDale
Neighborhood_BrkSide
Neighborhood_ClearCr
Neighborhood_CollgCr
Neighborhood_Crawfor
Neighborhood_Edwards
Neighborhood_Gilbert
Neighborhood_IDOTRR
Neighborhood_MeadowV
Neighborhood_Mitchel
Neighborhood_NAmes
Neighborhood_NPkVill
Neighborhood_NWAmes
Neighborhood_NoRidge
Neighborhood_NridgHt
Neighborhood_OldTown
Neighborhood_SWISU
Neighborhood_Sawyer
Neighborhood_SawyerW
Neighborhood_Somerst
Neighborhood_StoneBr
Neighborhood_Timber
Neighborhood_Veenker
Condition1_Artery
Condition1_Feedr
Condition1_Norm
Condition1_PosA
Condition1_PosN
Condition1_RRAe
Condition1_RRAn
Condition1_RRNe
Condition1_RRNn
Condition2_Artery
Condition2_Feedr
Condition2_Norm
Condition2_PosA
Condition2_PosN
Condition2_RRAe
Condition2_RRAn
Condition2_RRNn
BldgType_1Fam
BldgType_2fmCon
BldgType_Duplex
BldgType_Twnhs
BldgType_TwnhsE
HouseStyle_1.5Fin
HouseStyle_1.5Unf
HouseStyle_1Story
HouseStyle_2.5Fin
HouseStyle_2.5Unf
HouseStyle_2Story
HouseStyle_SFoyer
HouseStyle_SLvl
RoofStyle_Flat
RoofStyle_Gable
RoofStyle_Gambrel
RoofStyle_Hip
RoofStyle_Mansard
RoofStyle_Shed
RoofMatl_CompShg
RoofMatl_Membran
RoofMatl_Metal
RoofMatl_Roll
RoofMatl_Tar&Grv
RoofMatl_WdShake
RoofMatl_WdShngl
Exterior1st_AsbShng
Exterior1st_AsphShn
Exterior1st_BrkComm
Exterior1st_BrkFace
Exterior1st_CBlock
Exterior1st_CemntBd
Exterior1st_HdBoard
Exterior1st_ImStucc
Exterior1st_MetalSd
Exterior1st_Plywood
Exterior1st_Stone
Exterior1st_Stucco
Exterior1st_VinylSd
Exterior1st_WdSdng
Exterior1st_WdShing
Exterior2nd_AsbShng
Exterior2nd_AsphShn
Exterior2nd_BrkCmn
Exterior2nd_BrkFace
Exterior2nd_CBlock
Exterior2nd_CmentBd
Exterior2nd_HdBoard
Exterior2nd_ImStucc
Exterior2nd_MetalSd
Exterior2nd_Other
Exterior2nd_Plywood
Exterior2nd_Stone
Exterior2nd_Stucco
Exterior2nd_VinylSd
Exterior2nd_WdSdng
Exterior2nd_WdShng
MasVnrType_BrkCmn
MasVnrType_BrkFace
MasVnrType_None
MasVnrType_Stone
Foundation_BrkTil
Foundation_CBlock
Foundation_PConc
Foundation_Slab
Foundation_Stone
Foundation_Wood
Heating_Floor
Heating_GasA
Heating_GasW
Heating_Grav
Heating_OthW
Heating_Wall
Electrical_FuseA
Electrical_FuseF
Electrical_FuseP
Electrical_Mix
Electrical_SBrkr
GarageType_2Types
GarageType_Attchd
GarageType_Basment
GarageType_BuiltIn
GarageType_CarPort
GarageType_Detchd
GarageType_None
MiscFeature_Gar2
MiscFeature_None
MiscFeature_Othr
MiscFeature_Shed
MiscFeature_TenC
SaleType_COD
SaleType_CWD
SaleType_Con
SaleType_ConLD
SaleType_ConLI
SaleType_ConLw
SaleType_New
SaleType_Oth
SaleType_WD
SaleCondition_Abnorml
SaleCondition_AdjLand
SaleCondition_Alloca
SaleCondition_Family
SaleCondition_Normal
SaleCondition_Partial
PorchSF
BsmtFinSF

"""