#!/usr/bin/env python

import os,sys
from mvpa2.suite import *
from subprocess import call
import csv as csv
import shutil
import argparse


#parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i','--input',help="Manually input subject directory. If blank, run all subjects in folder.")
parser.add_argument('-csl','--cross_searchlight',help="Run across-classification searchlight only", action="store_true")
args = parser.parse_args()


#logging colors
sectionColor = "\033[94m"
sectionColor2 = "\033[96m"
groupColor = "\033[90m"
mainColor = "\033[92m"
pink = '\033[95m'
yellow = '\033[93m'
red = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

#Set up directorys and folders
basefolder = "/Volumes/MusicProject/MusicalBurst"
rois = ['firstlevel_run1.feat/mask.nii.gz','PlanumTemporale_ef.nii.gz']
instruments = ["all","violin","clarinet","voice"]

if args.input: 
    subjectfolders = []
    subjectfolders.append(args.input)
    ## show values ##
    print yellow + "Input file: %s%s" %(args.input,mainColor)
    #print subjectfolders

else:
    subjectfolders = [items for items in os.listdir(basefolder) if '9210' in items]
    subjectfolders.sort()

#subjectfolders = ['9195AD']
for subject in subjectfolders:
    print yellow + "Working on %s%s" %(subject,mainColor)
    datafile = "%s/%s/concatcopes.nii.gz" % (basefolder,subject)
    attributesfile = "%s/%s/attribute.txt" % (basefolder,subject)
    if not os.path.exists(attributesfile):
        shutil.copyfile("/Volumes/MusicProject/MusicalBurst/9151AD/attribute.txt",attributesfile)
        print 'Copying attributes file'
    allaccuracies = [];
    for roi in rois: 
        maskfile = "%s/%s/%s" % (basefolder,subject,roi)
        attr = SampleAttributes(attributesfile,header=['chunks','instrument','emotion'])
        dataset = fmri_dataset(datafile,targets=attr.emotion,chunks=attr.chunks,mask=maskfile) #Classifying emotion
        dataset.sa['instrument'] = attr.instrument

        for instr in instruments:
            if instr == "all":
                print sectionColor2 + "Classification of emotions for all instruments in %s%s" %(roi,mainColor) 
                ds = dataset.copy()
                #print ds.summary()
            else:
                print sectionColor2 + "Classification of emotions for %s only in %s%s" %(instr,roi,mainColor)
                ds = dataset[dataset.sa.instrument == instr]
                #print ds.summary()

            # Classification of emotions using Linear SVM
            splitter = NFoldPartitioner(cvtype=1) #partitioner to split training and testing datatasets, leave one out
            clf = LinearCSVMC() #classifier algorithm
            fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FractionTailSelector(0.1, mode='select', tail='upper')) #feature selection method
            fclf = FeatureSelectionClassifier(clf,fsel) #classifier that combines algorithm with feature selection
            if roi == 'firstlevel_run1.feat/mask.nii.gz':
                cvte = CrossValidation(fclf,splitter,errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']) #cross validation object
            else:
                cvte = CrossValidation(clf,splitter,errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']) #cross validation object
            
            cv_results = cvte(ds) #compute cross validation
            acc = np.mean(cv_results)
            print red + "%s accuracy: %.3f%s" %(instr,acc,mainColor)
            allaccuracies.append(acc)

            #Writing results to CSV File 
            # csvline = []
            # csvline.append(subject)
            # csvline.append(roi)
            # csvline.append('emotion')
            # csvline.append(instr)
            # csvline.append(acc)

            # resultfile = open(outfile,'a')
            # print sectionColor + "Writing results out to log file%s" %mainColor
            # #print csvline
            # writefile = csv.writer(resultfile)
            # writefile.writerow(csvline)
            # resultfile.close()

            #Searchlight
            if roi == 'firstlevel_run1.feat/mask.nii.gz':
                slfile = "%s/%s/searchlight_standard.nii.gz" % (basefolder,subject)
                ofile = "%s/%s/searchlight_%s.nii.gz" % (basefolder,subject,instr)
                if not os.path.exists(ofile):
                    cv = CrossValidation(clf,splitter,errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats'])
                    sl = sphere_searchlight(cv, radius=3, postproc=mean_sample(),nproc=6)
                    print sectionColor2 + "Performing %s searchligh for %s%s" % (instr,subject,mainColor)
                    res = sl(ds)

                    print sectionColor2 + "Converting file to nifti...%s" %(mainColor)
                    niftiresults = map2nifti(res, imghdr=dataset.a.imghdr)
                    niftiresults.to_filename(os.path.join(basefolder,subject,ofile))

                    outimage = "%s/%s/searchlight_standard_%s.nii.gz" % (basefolder,subject,instr)
                    mat =  "%s/%s/firstlevel_run1.feat/reg/example_func2standard.mat" % (basefolder,subject)
                    stand_image = "/usr/local/fsl/data/standard/MNI152_T1_2mm_brain"
                    command = "flirt -in %s -ref %s -applyxfm -init %s -o %s" % (ofile,stand_image,mat,outimage)
                    print sectionColor2 + 'Converting searchlight results to standard space%s'%(mainColor)
                    call(command, shell = True)
                else: 
                    print "%s already has %s" %(subject,ofile)

        ## Across classifications
        instrloop = ["clarinet","violin","voice","clarinet","voice","violin","clarinet"]
        print sectionColor2 + 'Working on across instrument classification%s' %(mainColor) 
        for counter in range(0,6):
            traininstr = instrloop[counter]
            testinstr = instrloop[counter+1]
            #print counter,traininstr,testinstr
            train_ds = dataset[dataset.sa.instrument == traininstr]
            test_ds = dataset[dataset.sa.instrument == testinstr]
            fsel.train(train_ds) #train feature selection on training set
            train_ds = fsel.forward(train_ds) #apply fsel to training set
            test_ds = fsel.forward(test_ds) #apply fsel to testing set too
            clf.train(train_ds)
            correctlist = clf.predict(test_ds.samples)==test_ds.targets
            acc = np.mean(correctlist)
            allaccuracies.append(acc)
            print "ROI: %s, Train: %s, Test: %s" % (roi,traininstr,testinstr)
            print red + "Accuracy: %.3f%s" %(acc,mainColor)

            # #Writing results to csv file
            # csvline = []
            # csvline.append(subject)
            # csvline.append(roi)
            # csvline.append('emotion')
            # traintest = '%s/%s' %(traininstr,testinstr)
            # csvline.append(traintest)
            # csvline.append(acc)
            # resultfile = open(outfile,'a')
            # print sectionColor + "Writing results out to log file%s" %mainColor
            # #print csvline
            # writefile = csv.writer(resultfile)
            # writefile.writerow(csvline)
            # resultfile.close()

    print subject
    for accuracy in allaccuracies:
        print accuracy

    print 










#get just clarinet, for example
#dataset = dataset[dataset.sa.instrument == 'clarinet'] #Accuracy is 0.33
 #Accuracy is 0.39
#dataset = dataset[dataset.sa.instrument == 'voice'] #Accuracy is 0.39

#Get just two emotions to look at the 2-way comparison
#dataset = dataset[dataset.targets != 'sad']
#print dataset.summary()
# dataset_s = dataset[dataset.targets == 'sad']
# dataset_test = dataset[dataset.targets.any(['happy','sad'])]
# dataset_test = dataset[(dataset.targets == 'happy') or (dataset.targets == 'sad')] 


#zscore(dataset, chunks_attr='chunks',dtype='float32')
#



#Get Confusion matrix 
cvte.ca.stats.plot(numbers=True)
pl.show()

#classifier warehouse
totry = clfs.warehouse.clfswh['binary']
for i in totry:                                      
    clf = i
    cvte = CrossValidation(clf,splitter,errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats'])
    cv_results = cvte(dataset)
    accuracy = np.mean(cv_results)
    print "Clf: %s accuracy: %.2f" % (clf.descr,accuracy)




######################################
## Across classifications Searchlight
######################################
#read in mask file
maskfile = "%s/%s/firstlevel_run1.feat/mask.nii.gz" %(basefolder,subject)
mask = fmri_dataset(maskfile,mask=maskfile)

#set up query engine to find indices of spheres
slradius = 4
space = 'voxel_indices'
kwa = {space: Sphere(slradius)}
qe = IndexQueryEngine(**kwa)
qe.train(mask)

#roi_ids are the sphere centers we want to test, i.e. all the voxels from the dataset
roi_ids = np.arange(mask.nfeatures)

#load in the training/testing data
dataset = fmri_dataset(datafile,targets=attr.emotion,chunks=attr.chunks,mask=maskfile)
dataset.sa['instrument'] = attr.instrument
data_clarinet = dataset[dataset.sa.instrument == 'clarinet'] #Accuracy is 0.33
data_violin = dataset[dataset.sa.instrument == 'violin'] #Accuracy is 0.39
data_voice = dataset[dataset.sa.instrument == 'voice'] #Accuracy is 0.39

#create blank array for results
results = np.zeros(mask.nfeatures)

#loop through each sphere center
sphere_accuracies = []
allsphereacc = np.array(("number","region","accuracies"))
for i, f in enumerate(roi_ids):

    print "Working on sphere %d of %d" % (i,mask.nfeatures)

    #get the voxel indices in a sphere from the query engine
    roi_fids = qe[f]
    #print roi_fids

    #select those voxels from the dataset
    sphere_data_train = data_clarinet[:,roi_fids]
    sphere_data_test = data_violin[:,roi_fids]
    clf = LinearCSVMC() 
    clf.train(sphere_data_train)
    correctlist = clf.predict(sphere_data_test)
    correctlist = clf.predict(sphere_data_test.samples)==sphere_data_test.targets
    accuracy = np.mean(correctlist)
    #print "Train on clarinet, test on violin: Accuracy: %.3f" % (accuracy)
    sphere_accuracies.append(accuracy)
    row = np.array((i,f,accuracy))
    allsphereacc = np.row_stack((allsphereacc,row))
    results = np.asarray(sphere_accuracies)
    #record result at same location in empty vector
    results[i] = accuracy


#save results to nifti image
outputfile = "sl_c_v.nii.gz"
outds = dataset.copy()
outds.samples = results  
niftiresults = map2nifti(outds, imghdr=mask.a.imghdr)
niftiresults.to_filename(outputfile)

##Classifying instruments 


    #dataset = fmri_dataset(datafile,targets=attr.instrument,chunks=attr.chunks,mask=maskfile) #Classifying instrument