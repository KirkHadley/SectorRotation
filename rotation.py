#mse = mean squared error
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy
from sklearn.ensemble import GradientBoostingRegressor
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from sklearn import preprocessing
from pybrain.structure.modules import TanhLayer, LinearLayer
from sklearn.linear_model import LinearRegression
import pickle
# list of vars
varsAndSectors=pd.read_csv("variables.csv")
varsAndSectors = varsAndSectors[:120:]
#main data set
data=pd.read_csv("data2.csv", index_col = 0)
data=data.apply(pd.Series.interpolate)
data.index = pd.to_datetime(data.index)
#test/validate data set
 
test=pd.read_csv("test.csv", index_col = 0)
test=test.apply(pd.Series.interpolate)
#this the newest data
# i.e. actual returns unknown
newpreds =pd.read_csv("fullpredictionset5.csv", index_col=0)
newpreds=newpreds.apply(pd.Series.interpolate)
 
#sometimes the dates come in wrong
data.index = pd.to_datetime(data.index)
test.index = pd.to_datetime(test.index)
newpreds.index = pd.to_datetime(newpreds.index)
sectors={"XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLB", "IYZ", "XLY", "XLU"}
called = dict()
 
for i in sectors:
    called[i] = varsAndSectors[varsAndSectors['Sector'] == i]
    called[i] = called[i]['Variables']
rfs= dict()
oob = dict()
fitted = dict()
importances =dict()
rgf = dict()
#train random forests
# hyper parameters for gbms and rfs
# were determined using grid search
for i in called:
    rgf[i] = RandomForestRegressor(n_estimators=7000, n_jobs=32, oob_score=True)
    rfs[i] = rgf[i].fit(data[called[i]], data[i])
    oob[i] = rfs[i].oob_score_
    fitted[i] = rfs[i].oob_prediction_
    importances[i] = rfs[i].feature_importances_
print oob
 
pickle.dump( importances, open( "rfimportances.p", "wb" ))
 
#to pandas
fitted = pd.DataFrame(fitted)
fitted.columns = fitted.columns + 'fitted'
for i in called:
    fitted[i] = data[i].values
fitted.to_csv('fitted.csv')
 
 
#predict valid/test set from forests trained above
predictedrf = dict()
mserf = dict()
for i in called:
    predictedrf[i] = rfs[i].predict(test[called[i]])        
    mserf[i] = numpy.mean((test[i] - predictedrf[i]))**2
predictedrf = pd.DataFrame(predictedrf)
predictedrf.columns = predictedrf.columns + 'predicted'
for i in called:
    predictedrf[i] = test[i].values
predictedrf.to_csv('predictionsrf.csv')
print mserf
 
 
#predict unknown data
newpredsrf = dict()
for i in called:
    newpredsrf[i] = rfs[i].predict(newpreds[called[i]])
newpredsrf = pd.DataFrame(newpredsrf)
newpredsrf.columns = newpredsrf.columns + 'RFNewpred'
for i in called:
    newpredsrf[i] = newpreds[i].values
newpredsrf.to_csv('newpredsdrf.csv')
 
# train gbms
# decided to use both b/c random forest catches regime change better than gbm
# but gbm models non-linearity more efficiently than rf
# unless learners in rf are deep, which defeats the purpose of weal learners
gbms= dict()
gbmfitted = dict()
gbmimportances=dict()
gbm = dict()
for i in called:
    gbm[i] = GradientBoostingRegressor(n_estimators=7000, learning_rate=.001, max_features=.33,max_depth=3, subsample=.6,random_state=0, loss='ls')
    gbms[i] = gbm[i].fit(data[called[i]], data[i])
msefittedgbm = dict()    
#get gbm fits on trained set
for i in called:
    gbmfitted[i] = gbms[i].predict(data[called[i]])
    gbmimportances[i] = gbms[i].feature_importances_
    msefittedgbm[i] = numpy.mean((data[i] - gbmfitted[i]))**2
print msefittedgbm
pickle.dump(gbmimportances, open( "rfimportances.p", "wb" ))
# into pandas
gbmfitted = pd.DataFrame(gbmfitted)
gbmfitted.columns = gbmfitted.columns + 'gbmfitted'
for i in called:
    gbmfitted[i] = data[i].values
gbmfitted.to_csv('fittedgbm.csv')
# predicting valid/train w/ gbm
predictedgbm = dict()
msegbm = dict()
for i in called:
    predictedgbm[i] = gbm[i].predict(test[called[i]])        
    msegbm[i] = numpy.mean((test[i] - predictedgbm[i]))**2
predictedgbm = pd.DataFrame(predictedgbm)
predictedgbm.columns = predictedgbm.columns + 'predicted'
for i in called:
    predictedgbm[i] = test[i].values
predictedgbm.to_csv('predictionsgbm.csv')
 
 
# predicting unknwown data w/ gbm
newpredgbm= dict()
for i in called:
    newpredgbm[i] = gbm[i].predict(newpreds[called[i]])
newpredgbm = pd.DataFrame(newpredgbm)
newpredgbm.columns = newpredgbm.columns + 'predicted'
for i in called:
    newpredgbm[i] = newpreds[i].values
newpredgbm.to_csv('newpredgbm.csv')
 
 
 
 
 
 
 
#nn
varsAndSectors=pd.read_csv("variables.csv")
sectors={"XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLB", "IYZ", "XLY", "XLU"}
called = dict()
for i in sectors:
    called[i] = varsAndSectors[varsAndSectors['Sector'] == i]
    called[i] = called[i]['Variables']
#pybrain is picky about data input format so reformatting here
# also preprocessing
inputArrays = dict()
yArrays = dict()
for i in called:
    inputArrays[i] = data[called[i]].values
    inputArrays[i] = preprocessing.scale(inputArrays[i])
    yArrays[i] = data[i].values
    yArrays[i] = yArrays[i].reshape(-1,1)
input_size = dict()
target_size = dict()
ds = dict()
for i in inputArrays:
    input_size[i] = inputArrays[i].shape[1]
    target_size[i] = yArrays[i].shape[1]
    ds[i] = SDS( input_size[i], target_size[i])
    ds[i].setField( 'input', inputArrays[i])
    ds[i].setField( 'target', yArrays[i] )
# doing the same for the test/validate set
testinputArrays = dict()
testyArrays = dict()
for i in called:
    testinputArrays[i] = test[called[i]].values
    testinputArrays[i] = preprocessing.scale(testinputArrays[i])
    testyArrays[i] = test[i].values
    testyArrays[i] = testyArrays[i].reshape(-1,1)
testinput_size = dict()
testtarget_size = dict()
testds = dict()
for i in testinputArrays:
    testinput_size[i] = testinputArrays[i].shape[1]
    testtarget_size[i] = testyArrays[i].shape[1]
    testds[i] = SDS(testinput_size[i], testtarget_size[i])
    testds[i].setField('input', testinputArrays[i])
    testds[i].setField('target', testyArrays[i])
 
#now for the unknown set
newpredsinputArrays = dict()
newpredsyArrays = dict()
for i in called:
    newpredsinputArrays[i] = newpreds[called[i]].values
    newpredsinputArrays[i] = preprocessing.scale(newpredsinputArrays[i])
    newpredsyArrays[i] = newpreds[i].values
    newpredsyArrays[i] = newpredsyArrays[i].reshape(-1,1)
newpredsinput_size = dict()
newpredstarget_size = dict()
newpredsds = dict()
for i in newpredsinputArrays:
    newpredsinput_size[i] = newpredsinputArrays[i].shape[1]
    newpredstarget_size[i] = newpredsyArrays[i].shape[1]
    newpredsds[i] = SDS( newpredsinput_size[i], newpredstarget_size[i])
    newpredsds[i].setField( 'input', newpredsinputArrays[i])
    newpredsds[i].setField( 'target', newpredsyArrays[i] )
 
# function to create nn for a given sector, learn it on the training set
# and predict on the test set
# the nn is trained on the same inputs as above
# as well as standard error of a 6-month ahead arima prediction
# ^ necessary b/c will be making predictions w/o knowing forward rolling 6 mo returns
# i.e. for predictions made in may, only have complete arima set through sept
def networkbuild(x, ds, testds, newpredsds):
    net = buildNetwork(13, 30, 1, hiddenclass = TanhLayer, outclass=LinearLayer)
    preds = dict()
    sector = dict()    
    testpreds = dict()
    testsector = dict()
    newnnpreds = dict()
    trainer = BackpropTrainer( net,ds[x],momentum=0.01,verbose=True, learningrate=.001)
    trainer.trainEpochs(500)
    predictions = net.activateOnDataset( ds[x] )
    for i in range(0, len(predictions)):
        preds[i]=predictions[i][0]
    preds = preds.values()
    for i in range(0, len(ds[x])):
        sector[i] = ds[x]['target'][i][0]
    sector= sector.values()
    testpredictions = net.activateOnDataset( testds[x] )
    for i in range(0, len(testpredictions)):
        testpreds[i]=testpredictions[i][0]
    testpreds = testpreds.values()
    for i in range(0, len(testds[x])):
        testsector[i] = testds[x]['target'][i][0]
    testsector= testsector.values()
    newnnpredictions= net.activateOnDataset( newpredsds[x] )
    for i in range(0, len(newnnpredictions)):
        newnnpreds[i]=newnnpredictions[i][0]
    newnnpreds = newnnpreds.values()
    everything = {'sector': x,
    'inputs':ds[x]['input'],
    'network': net,
    'trainer': trainer,
    'results':preds,
    'testresults':testpreds,
    'newpredictions':newnnpreds}
    return everything
nets = dict()
results = dict()
testresults = dict()
newpredresults = dict()
# running function above for all sectors
for i in called:
    nets[i] = networkbuild(i, ds, testds, newpredsds)
    results[i] = nets[i]['results']
    testresults[i] = nets[i]['testresults']
    newpredresults[i] = nets[i]['newpredictions']
# into pandas
results = pd.DataFrame(results)
results.columns = results.columns + 'PredictedNN'    
testresults = pd.DataFrame(testresults)
testresults.columns = testresults.columns + 'testPredictedNN'
newpredresults = pd.DataFrame(newpredresults)
newpredresults.columns = newpredresults.columns + 'testPredictedNN'
 
#linear ensemble of the four models
# nn, gbm, rf, and arima
# remember arima created in R (better implementation)
# arima predictions were included in both the
# training and test data sets
gbmfit = gbmfitted
gbmpred = predictedgbm 
rfFit = fitted
rfpred = predictedrf
gbmfit= gbmfit.ix[:,0:10]
fitactual = rfFit.ix[:,10:21]
rfFit = rfFit.ix[:,0:10]
dates = data.index
gbmfit['dates'] = dates
rfFit['dates'] = dates
fitactual['dates'] = dates
results['dates'] = dates
full = pd.merge(gbmfit, rfFit)
full = pd.merge(full, fitactual)
full = pd.merge(full, fitactual)
full = pd.merge(full, results)
full = full.reindex_axis(sorted(full.columns), axis=1)
full.set_index(dates, inplace=True)
 
 
 
 
 
gbmpred = gbmpred.ix[:,0:10]
predactual = rfpred.ix[:,10:21]
rfpred = rfpred.ix[:,0:10]
rfpred.columns = rfpred.columns + 'rf'
gbmpred.columns = gbmpred.columns + 'gbm'
dates1 = test.index
gbmpred['dates'] = dates1
rfpred['dates'] = dates1
predactual['dates'] = dates1
testresults['dates'] = dates1
fullpred = pd.merge(gbmpred, rfpred)
fullpred = pd.merge(fullpred, predactual)
fullpred = pd.merge(fullpred, testresults)
fullpred = fullpred.reindex_axis(sorted(fullpred.columns), axis=1)
fullpred.set_index(dates1, inplace=True)
 
 
 
 
 
newpredgbm =  newpredgbm.ix[:,0:10]
newpredsrf = newpredsrf.ix[:,0:10]
dates2 = newpreds.index
newpredgbm['dates'] = dates2
newpredsrf['dates'] = dates2
newpredresults['dates'] = dates2
newpredsrf.columns = rfpred.columns
newpredgbm.columns = gbmpred.columns
fullnewpred = pd.merge(newpredgbm, newpredsrf)
fullnewpred = pd.merge(fullnewpred, newpredresults)
newpreds['dates'] = dates2
fullnewpred = pd.merge(fullnewpred, newpreds)
fullnewpred = fullnewpred.reindex_axis(sorted(fullnewpred.columns), axis=1)
fullnewpred.set_index(dates2, inplace=True)
 
 
 
 
 
errornames = ["XLKpredictions", "XLFpredictions", "XLEpredictions", "XLVpredictions", "XLIpredictions", "XLPpredictions", "XLBpredictions", "IYZpredictions", "XLYpredictions", "XLUpredictions"]
for i in range(0,len(errornames)):
    full[errornames[i]] = data[errornames[i]]
full = full.reindex_axis(sorted(full.columns), axis=1)
for i in range(0,len(errornames)):
    fullpred[errornames[i]] = test[errornames[i]]
fullpred = fullpred.reindex_axis(sorted(fullpred.columns), axis=1)
sectornames=["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLB", "IYZ", "XLY", "XLU"]
linears = dict()
for i in range(0,len(sectornames)):
    linears[sectornames[i]] = {'rf':sectornames[i] + 'predictedrf', 'gbm':sectornames[i] + 'predictedgbm', 'arima': sectornames[i] + 'predictions', 'nn':sectornames[i] + 'testPredictedNN'}
regr = dict()
lms = dict()
regr1 = dict()
lms1 = dict()
linearvalids = dict()
newlinears = dict()
firstpredsmodel = dict()
firstcoeffs = dict()
for i in linears:
    regr[i] = LinearRegression()
    lms[i] = regr[i].fit(fullpred[linears[i].values()], fullpred[i])
    linearvalids[i] = lms[i].predict(fullpred[linears[i].values()])
    newlinears[i] = lms[i].predict(fullnewpred[linears[i].values()])
 
msepred = dict()
for i in linears:
    msepred[i] = numpy.mean((fullpred[i] - linearvalids[i]))**2
for i in range(0,len(sectornames)):
    firstcoeffs[sectornames[i]] = {'rf':sectornames[i] + 'fitted', 'gbm':sectornames[i] + 'gbmfitted', 'arima': sectornames[i] + 'predictions', 'nn':sectornames[i] + 'PredictedNN'}
for i in linears:
    regr1[i] = LinearRegression()
    lms1[i] = regr1[i].fit(full[firstcoeffs[i].values()], full[i])
    firstpredsmodel[i] = lms1[i].predict(full[firstcoeffs[i].values()])
 
msefull = dict()
for i in linears:
    msefull[i] = numpy.mean((full[i] - firstpredsmodel[i]))**2
linearvalids= pd.DataFrame(linearvalids)
linearvalids.columns = linearvalids.columns + 'linear'
linearvalids['dates'] = dates1
fullpred = pd.merge(fullpred, linearvalids)
fullpred = fullpred.reindex_axis(sorted(fullpred.columns), axis=1)
fullpred.to_csv('validation.csv')
firstpredsmodel= pd.DataFrame(firstpredsmodel)
firstpredsmodel.columns = firstpredsmodel.columns + 'linear'
firstpredsmodel['dates'] = dates
full = pd.merge(full, firstpredsmodel)
full= full.reindex_axis(sorted(full.columns), axis=1)
full.to_csv('trainedfits.csv')
 
newlinears = pd.DataFrame(newlinears)
newlinears.columns = newlinears.columns + 'linear'
newlinears['dates'] = dates2
fullnewpred = pd.merge(fullnewpred, newlinears)
fullnewpred = fullnewpred.reindex_axis(sorted(fullnewpred.columns), axis=1)
fullnewpred.to_csv('newpred.csv')
