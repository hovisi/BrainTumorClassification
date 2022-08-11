#import given files now
from dataset import Dataset



#Notation Note: poisonous (pois) and edible (ed)

#Declare and create finctions 

#################################################
#This function will train the model 
#inputs: D= Dataset m= m-estimate
#outs: Probability Dictionary for model
##################################################
def trainModel(D , m):
    #create dictionary for probabilities
    #going to do embadded dictionaries in order to create the probability tables
    #Format of dictionary: layer atrribute -> category of attribute --> class (ed or pois)
    ProbabilitiesFinal = {} 
    
    for attribute in D.attributes: #go through each attribute
        attributeDic = {} #clear, so ready for next attribute
        
        for cat in D.getAttributeValues(attribute): #go through each category in each attribute
            CategoryDic = {} #clear, so ready for next category of attribute
            
            for classification in D.getAttributeValues("class"): #go through each classifification (pois or ed) for each category in each attribute

                #now find the probability of this case occuring
                probCase =  (len(D.selectSubset({attribute: cat , "class": classification}))+ m * (1/len(D.getAttributeValues(attribute)))) / (m +len(D.selectSubset({"class": classification})))
                
                CategoryDic.update({classification : probCase}) #add class dictionary to the category Dictionary
        
            attributeDic.update({cat : CategoryDic}) #add category dictionary to the attribute Dictionary

        ProbabilitiesFinal.update({attribute : attributeDic}) #add attribute dictionary to the final probabilities Dictionary
    
    #return Probabilities Dictionary
    return ProbabilitiesFinal

############################################################
#This function will calculate the accuracy of the model
#inputs: D= Dataset m= m-estimate ProbabilitiesFinal= Dictionary of trained model
#outs: Probability Dictionary for model
##############################################################
def caluateAccuracy(D , m, ProbabilitiesFinal):
    correctlypredicted = 0 #initalize correctly predicted after each instance
    numberEd = len(D.selectSubset({"class":"e"}))
    numberPois = len(D.selectSubset({"class":"p"}))
    numberMush = numberEd + numberPois

    #go through each instance and keep track of the index 
    for index in range(len(D.instances)):  
        #initalize probabilities
        probPois = (numberPois +  m * (1/2))/(numberMush+ m) 
        probEd = (numberEd +  m * (1/2))/(numberMush+ m)

        #go through each attribute
        for attribute in D.attributes: 
            #want to exclude class attribute
            if(attribute != "class"): 
                cat = D.getInstanceValue(attribute, index)
                probPois = probPois * ProbabilitiesFinal[attribute][cat]["p"] 
                probEd = probEd * ProbabilitiesFinal[attribute][cat]["e"] 

        #normalize
        NormPois = (probPois/(probPois+probEd)) * 100
        NormEd = (probEd/(probPois+probEd)) * 100

        #now see if correctly predicted the mushroom corrctly
        #correctly predicted if higher percentage to accurate class for training set, if 50/50 then tie goes to pois
        if(NormEd > NormPois): 
            PredEdible = True; 
        else: 
            PredEdible = False; 

        cat = D.getInstanceValue("class", index)

        if(cat== "p" and PredEdible == False): 
            correctlypredicted = correctlypredicted +1
    
        if(cat == "e" and PredEdible == True): 
            correctlypredicted = correctlypredicted +1
    
    #output accuraccy of classifying testing data
    percentageCorrectPredict = (correctlypredicted/len(D.instances)) * 100

    return percentageCorrectPredict


#set up file variables
fileTest =  "mushroom-testing.data"
fileTrain =  "mushroom-training.data" 

#use constructor, to set up data
D = Dataset(fileTrain) #training dataset
T = Dataset(fileTest) #testing dataset

#declare the different m-values waanted to be used
mVals = (0, .5 , 1,  2, 10)

#create loop that runs through m-values and trainsand gets accuracy of model
for m in mVals: 
    #declare and reset trained dictionary and percentage accuracy
    ProbabilitiesFinal = {}
    percentageCorrectPredict = 0

    # train the model with the test data set
    ProbabilitiesFinal = trainModel(D , m) 

    #calculate accuracy of the model for the training set
    percentageCorrectPredict= caluateAccuracy(D, m, ProbabilitiesFinal)
    print("This Naive Bayes Classifier has a classification accuracy for the training data of" ,  percentageCorrectPredict , "%" , "where m=", m)

    #calculate accuracy of the model for the training set
    percentageCorrectPredict= caluateAccuracy(T, m, ProbabilitiesFinal)
    print("This Naive Bayes Classifier has a classification accuracy for the test data of" ,  percentageCorrectPredict , "%" , "where m=", m)