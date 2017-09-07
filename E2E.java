package structuredPredictionNLG;

import com.google.common.collect.Lists;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.mt.metrics.BLEUMetric;
import edu.stanford.nlp.mt.tools.NISTTokenizer;
import edu.stanford.nlp.mt.util.IString;
import edu.stanford.nlp.mt.util.IStrings;
import edu.stanford.nlp.mt.util.ScoredFeaturizedTranslation;
import edu.stanford.nlp.mt.util.Sequence;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import imitationLearning.JLOLS;
import jarow.Instance;
import jarow.JAROW;
import jarow.Prediction;
import java.util.List;
import similarity_measures.Levenshtein;
import similarity_measures.Rouge;
import simpleLM.SimpleLM;

class ValueComparator implements Comparator<String> {

    Map<String, Integer> base;

    public ValueComparator(HashMap<String, Integer> wordDictionary) {
        this.base = wordDictionary;
    }

    // Note: this comparator imposes orderings that are inconsistent with
    // equals.
    public int compare(String a, String b) {
        if (base.get(a) >= base.get(b)) {
            return -1;
        } else {
            return 1;
        } // returning 0 would merge keys
    }
}

public class E2E extends DatasetParser {

    final String singlePredicate = "#PREDICATE";
    private JLOLS ILEngine;

    public E2E(String[] args) {
        super(args);

    }

    public static void main(String[] args) {
        E2E e2e = new E2E(args);
        e2e.parseDataset();
        e2e.ILEngine = new JLOLS(e2e);
        JLOLS.targettedExploration = 10;
        JLOLS.batchSize = 1000;
        long start_time = System.currentTimeMillis();

        e2e.createTrainingData();
        long end_time = System.currentTimeMillis();
        long running_time = end_time - start_time;
        System.out.println("running time is " + running_time);
        e2e.performImitationLearning(e2e.ILEngine);
    }

    @Override
    public void parseDataset() {
        File trainingDataFile = new File("e2e_traindev/trainset.csv");
        File devDataFile = new File("e2e_traindev/devset.csv");
        // Initialize the collections
        setPredicates(new ArrayList<>());
        setAttributes(new HashMap<>());
        setAttributeValuePairs(new HashMap<>());
        setValueAlignments(new HashMap<>());
        getAttributes().put(singlePredicate, new HashSet<>());
        getDatasetInstances().put(singlePredicate, new ArrayList<>());

        if (isResetStoredCaches() || !loadLists()) {
            createLists(trainingDataFile);
            getTrainingData().addAll(getDatasetInstances().get(singlePredicate)
            //.subList(0, 1000)
            );
            getDatasetInstances().put(singlePredicate, new ArrayList<>());
            createLists(devDataFile);
            //Collections.shuffle(getDatasetInstances().get(singlePredicate), new Random(123));
            getValidationData().addAll(getDatasetInstances().get(singlePredicate)
            //.subList(0, 100)
            );
            
            // Create the refs for DEV data, as described in https://github.com/tuetschek/e2e-metrics/tree/master/example-inputs
            for (DatasetInstance di : getValidationData()) {
                HashSet<String> refs = new HashSet<>();
                for (DatasetInstance di2 : getValidationData()) {
                    if (di2.getMeaningRepresentation().getMRstr().equals(di.getMeaningRepresentation().getMRstr())) {
                        refs.add(di2.getDirectReference());
                    }
                }
                di.setEvaluationReferences(refs);
            }

            getDatasetInstances().get(singlePredicate).addAll(getTrainingData());
            writeLists();
        }

        // Dataset analysis and re-splitting
        /*HashMap<String, String> uniqueValMRsTrain = new HashMap<>();
        HashMap<String, String> distinctValMRsTrain = new HashMap<>();
        for (DatasetInstance val : getTrainingData()) {
            boolean distinct = true;
            if (!uniqueValMRsTrain.containsKey(val.getMeaningRepresentation().getAbstractMR())) {
                String evalRefs = "";
                boolean isFirst = true;
                for (String r : val.getEvaluationReferences()) {
                    if (isFirst) {                        
                        evalRefs += r;
                    } else {
                        evalRefs += " | " + r;
                    }
                    isFirst = false;
                }
                evalRefs = evalRefs.trim();
                uniqueValMRsTrain.put(val.getMeaningRepresentation().getAbstractMR(), "\"" + val.getMeaningRepresentation().getMRstr() + "\"," + evalRefs);
            }
            for (DatasetInstance tr : getValidationData()) {
                if (tr.getMeaningRepresentation().getAbstractMR().equals(val.getMeaningRepresentation().getAbstractMR())) {
                //if (tr.getMeaningRepresentation().getAttributeValues().keySet().equals(val.getMeaningRepresentation().getAttributeValues().keySet())) {
                    distinct = false;
                }
            }
            if (distinct) {
                System.out.println(val.getMeaningRepresentation().getAttributeValues().keySet());
                System.out.println("\"" + val.getMeaningRepresentation().getMRstr() + "\"," + val.getDirectReference());
                if (!distinctValMRsTrain.containsKey(val.getMeaningRepresentation().getAbstractMR())) {
                    String evalRefs = "";
                    boolean isFirst = true;
                    for (String r : val.getEvaluationReferences()) {
                        if (isFirst) {                        
                            evalRefs += r;
                        } else {
                            evalRefs += " | " + r;
                        }
                        isFirst = false;
                    }
                    evalRefs = evalRefs.trim();
                    distinctValMRsTrain.put(val.getMeaningRepresentation().getAbstractMR(), "\"" + val.getMeaningRepresentation().getMRstr() + "\"," + evalRefs);
                }
            }
        }
        
        HashMap<String, String> uniqueValMRsDev = new HashMap<>();
        HashMap<String, String> distinctValMRsDev = new HashMap<>();
        for (DatasetInstance val : getValidationData()) {
            boolean distinct = true;
            if (!uniqueValMRsDev.containsKey(val.getMeaningRepresentation().getAbstractMR())) {
                String evalRefs = "";
                boolean isFirst = true;
                for (String r : val.getEvaluationReferences()) {
                    if (isFirst) {                        
                        evalRefs += r;
                    } else {
                        evalRefs += " | " + r;
                    }
                    isFirst = false;
                }
                evalRefs = evalRefs.trim();
                uniqueValMRsDev.put(val.getMeaningRepresentation().getAbstractMR(), "\"" + val.getMeaningRepresentation().getMRstr() + "\"," + evalRefs);
            }
            for (DatasetInstance tr : getTrainingData()) {
                if (tr.getMeaningRepresentation().getAbstractMR().equals(val.getMeaningRepresentation().getAbstractMR())) {
                //if (tr.getMeaningRepresentation().getAttributeValues().keySet().equals(val.getMeaningRepresentation().getAttributeValues().keySet())) {
                    distinct = false;
                }
            }
            if (distinct) {
                System.out.println(val.getMeaningRepresentation().getAttributeValues().keySet());
                System.out.println("\"" + val.getMeaningRepresentation().getMRstr() + "\"," + val.getDirectReference());
                if (!distinctValMRsDev.containsKey(val.getMeaningRepresentation().getAbstractMR())) {
                    String evalRefs = "";
                    boolean isFirst = true;
                    for (String r : val.getEvaluationReferences()) {
                        if (isFirst) {                        
                            evalRefs += r;
                        } else {
                            evalRefs += " | " + r;
                        }
                        isFirst = false;
                    }
                    evalRefs = evalRefs.trim();
                    distinctValMRsDev.put(val.getMeaningRepresentation().getAbstractMR(), "\"" + val.getMeaningRepresentation().getMRstr() + "\"," + evalRefs);
                }
            }
        }
        HashMap<Integer, HashMap<String, String>> uniqueValMRsAll = new HashMap<>();
        ArrayList<DatasetInstance> all = new ArrayList<>();
        all.addAll(getTrainingData());
        all.addAll(getValidationData());
        for (DatasetInstance val : all) {
            if (!uniqueValMRsAll.containsKey(val.getMeaningRepresentation().getAbstractMR())) {
                String evalRefs = "";
                boolean isFirst = true;
                for (String r : val.getEvaluationReferences()) {
                    if (isFirst) {                        
                        evalRefs += r;
                    } else {
                        evalRefs += " | " + r;
                    }
                    isFirst = false;
                }
                evalRefs = evalRefs.trim();
                if (!uniqueValMRsAll.containsKey(val.getMeaningRepresentation().getAttributeValues().keySet().size())) {
                    uniqueValMRsAll.put(val.getMeaningRepresentation().getAttributeValues().keySet().size(), new HashMap<>());
                }
                uniqueValMRsAll.get(val.getMeaningRepresentation().getAttributeValues().keySet().size()).put(val.getMeaningRepresentation().getAbstractMR(), "\"" + val.getMeaningRepresentation().getMRstr() + "\"," + evalRefs);
            }
        }
        HashMap<String, String> uniqueValMRsNewTrain = new HashMap<>();
        HashMap<String, String> uniqueValMRsNewDev = new HashMap<>();        
        for (Integer s : uniqueValMRsAll.keySet()) {
            ArrayList<String> keys = new ArrayList<String>(uniqueValMRsAll.get(s).keySet());
            Collections.shuffle(keys, new Random(123));
            
            int split = ((Double) java.lang.Math.floor(keys.size() * 0.9)).intValue();
            for (String k : keys.subList(0, split)) {
                uniqueValMRsNewTrain.put(k, uniqueValMRsAll.get(s).get(k));
            }
            for (String k : keys.subList(split, keys.size())) {
                uniqueValMRsNewDev.put(k, uniqueValMRsAll.get(s).get(k));
            }
        }
        try {
            java.io.PrintWriter outT = new java.io.PrintWriter(new java.io.FileWriter("unique_trainset.csv"));
            outT.println("mr,ref");
            for (String s : uniqueValMRsTrain.values()) {
                outT.println(s);
            }
            outT.flush();
            outT.close();
            
            java.io.PrintWriter out = new java.io.PrintWriter(new java.io.FileWriter("unique_devset.csv"));
            out.println("mr,ref");
            for (String s : uniqueValMRsDev.values()) {
                out.println(s);
            }
            out.flush();
            out.close();
            
            java.io.PrintWriter out2 = new java.io.PrintWriter(new java.io.FileWriter("distinct_devset.csv"));
            out2.println("mr,ref");
            for (String s : distinctValMRsDev.values()) {
                out2.println(s);
            }
            out2.flush();
            out2.close();

            java.io.PrintWriter out3 = new java.io.PrintWriter(new java.io.FileWriter("uniqueAndDistnct_newTrainSet.csv"));
            out3.println("mr,ref");
            for (String s : uniqueValMRsNewTrain.values()) {
                out3.println(s);
            }
            out3.flush();
            out3.close();

            java.io.PrintWriter out4 = new java.io.PrintWriter(new java.io.FileWriter("uniqueAndDistnct_newDevSet.csv"));
            out4.println("mr,ref");
            for (String s : uniqueValMRsNewDev.values()) {
                out4.println(s);
            }
            out4.flush();
            out4.close();
        } catch (IOException ex) {
            Logger.getLogger(E2E.class.getName()).log(Level.SEVERE, null, ex);
        }

        System.out.println(uniqueValMRsNewTrain.keySet().size() + " / " + all.size());
        System.out.println(uniqueValMRsNewDev.keySet().size() + " / " + all.size());
        System.out.println(uniqueValMRsTrain.keySet().size() + " / " + getTrainingData().size());
        System.out.println(distinctValMRsTrain.keySet().size() + " / " + getTrainingData().size());
        System.out.println(uniqueValMRsDev.keySet().size() + " / " + getValidationData().size());
        System.out.println(distinctValMRsDev.keySet().size() + " / " + getValidationData().size());
        System.exit(0);*/
        System.out.println("Training data size: " + getTrainingData().size());
        System.out.println("Validation data size: " + getValidationData().size());

    }

    public void createLists(File dataFile) {
        System.out.println("create lists");

        ArrayList<String> dataPart = new ArrayList<>();

        //we read in the data from the data files.
        try {
            BufferedReader br = new BufferedReader(new FileReader(dataFile));

            String s;
            try {
                s = br.readLine();
                while (s != null) {
                    dataPart.add(s);
                    s = br.readLine();
                }

            } catch (IOException e) {

                e.printStackTrace();
            }
            try {
                br.close();
            } catch (IOException e) {

            }
        } catch (FileNotFoundException e) {

            e.printStackTrace();
        }
        // in this data set  we don't have predicate so set a single predicate just for populate the data structure
        getPredicates().add(singlePredicate);
        // remove the column name line
        dataPart.remove(0);
        // fix the errors in the data set
        for (int i = 0; i < dataPart.size(); i++) {

            if (!dataPart.get(i).contains("\",")) {
                String line = dataPart.get(i - 1);
                line = line + dataPart.get(i);
                dataPart.remove(i);
            }
        }
        // for each instance    
        int num = 0;
        for (String line : dataPart) {

            //System.out.println(num);
            num++;

            String MRPart = line.split("\",")[0];
            String RefPart = line.split("\",")[1].toLowerCase();
            if (RefPart.equals("café")) {
                continue;
            }
            if (MRPart.startsWith("\"")) {
                MRPart = MRPart.substring(1);
            }
            if (RefPart.startsWith("\"")) {
                RefPart = RefPart.substring(1);
            }
            if (RefPart.endsWith("\"")) {
                RefPart = RefPart.substring(0, RefPart.length() - 1);
            }
            String[] MRs = MRPart.split(",");
            // original value to delexicalized value
            HashMap<String, String> delexicalizedMap = new HashMap<>();
            // each instance's attribute value pairs
            HashMap<String, HashSet<String>> attributeValues = new HashMap<>();
            // for each attribute value pairs
            for (String mr : MRs) {
                String value = mr.substring(mr.indexOf("[") + 1, mr.indexOf("]")).trim().toLowerCase();
                String attribute = mr.substring(0, mr.indexOf("[")).trim().toLowerCase();
                if (attribute.equals("name")) {
                    // delexicalize name values
                    String delexValue = Action.TOKEN_X + attribute + "_0";
                    delexicalizedMap.put(delexValue, value);
                    value = delexValue;

                }
                if (attribute.equals("near")) {
                    //delexicalize near values
                    String delexValue = Action.TOKEN_X + attribute + "_0";
                    delexicalizedMap.put(delexValue, value);
                    value = delexValue;
                }
                if (value.equals("yes") || value.equals("no")) {

                    value = attribute + "_" + value;
                }
                getAttributes().put(singlePredicate, new HashSet<>());
                if (attribute != null) {
                    getAttributes().get(singlePredicate).add(attribute);
                    if (!getAttributeValuePairs().containsKey(attribute)) {
                        getAttributeValuePairs().put(attribute, new HashSet<String>());
                    }
                    if (!attributeValues.containsKey(attribute)) {
                        attributeValues.put(attribute, new HashSet<>());
                    }
                    if (value != null) {
                        getAttributeValuePairs().get(attribute).add(value);
                        attributeValues.get(attribute).add(value);
                    }
                }
            }

            //RefPart = " "+RefPart+" ";
            for (String deValue : delexicalizedMap.keySet()) {
                String value = delexicalizedMap.get(deValue);
                if (RefPart.contains(value)) {
                    RefPart = RefPart.replace(value, deValue).trim();
                }
            }

            //if(!RefPart.contains("@x@name_0")&&!RefPart.contains("@x@near_0")){
            //System.out.println(RefPart);
            //}
            /*
        	if(RefPart.contains("name-")){
        		RefPart = RefPart.replace("name-","name -");
        	}
        	if(RefPart.contains("@x@names")){
        		RefPart = RefPart.replace("@x@names", "@x@name s");
        	}
        	if(RefPart.contains("@x@nearian")){
        		RefPart = RefPart.replace("@x@nearian", "@x@near");
        	}
        	if(RefPart.contains("@x@namen")){
        		RefPart = RefPart.replace("@x@namen", "@x@name");
        	}
        	if(RefPart.contains("@x@nearn")){
        		RefPart = RefPart.replace("@x@nearn", "@x@near");
        	}
        	if(RefPart.contains("@x@nears")){
        		RefPart = RefPart.replace("@x@nears", "@x@near s");
        	}
        	if(RefPart.contains("@x@name_0s")){
        		RefPart = RefPart.replace("@x@name_0s", "@x@name_0 s");
        	}
        	if(RefPart.contains("@x@near_0s")){
        		RefPart = RefPart.replace("@x@near_0s", "@x@near_0 s");
        	}*/
            // create MR for each instance 
            //MeaningRepresentation MR = new MeaningRepresentation(singlePredicate,attributeValues,MRPart,delexicalizedMap);
            // start create the value alignments
            ArrayList<String> observedAttrValueSequence = new ArrayList<>();
            ArrayList<String> observedWordSequence = new ArrayList<>();
            // replace the punctuation in the reference
            //RefPart = RefPart.replaceAll("[.,?:;!'-]", " "+Action.TOKEN_PUNCT+" ");
            //String[] words = RefPart.replaceAll("[.,?:;!'-]", " "+Action.TOKEN_PUNCT+" ").split(" ");
            String[] words = RefPart.replace(", ,", " , ").replace(". .", " . ").replaceAll("[.,?:;!'-]", " $0 ").split("\\s+");
            for (String w : words) {
                if (w.contains("0f")) {
                    w = w.replace("0f", "of");
                }

                Pattern p1 = Pattern.compile("([0-9]+)([a-z]+)");
                Matcher m1 = p1.matcher(w);
                Pattern p2 = Pattern.compile("([a-z]+)([0-9]+)");
                Matcher m2 = p2.matcher(w);
                Pattern p3 = Pattern.compile("(£)([a-z]+)");
                Matcher m3 = p3.matcher(w);
                Pattern p4 = Pattern.compile("([a-z]+)(£[0-9]+)");
                Matcher m4 = p4.matcher(w);
                Pattern p5 = Pattern.compile("([0-9]+)([a-z]+)([0-9]+)");
                Matcher m5 = p5.matcher(w);
                Pattern p6 = Pattern.compile("([0-9]+)(@x@[a-z]+_0)");
                Matcher m6 = p6.matcher(w);
                if (m1.find()) {
                    observedWordSequence.add(m1.group(1).trim());
                    observedWordSequence.add(m1.group(2).trim());

                } else if (m2.find()) {

                    observedWordSequence.add(m2.group(1).trim());
                    observedWordSequence.add(m2.group(2).trim());

                } else if (m3.find()) {

                    observedWordSequence.add(m3.group(1).trim());
                    observedWordSequence.add(m3.group(2).trim());

                } else if (m4.find()) {

                    observedWordSequence.add(m4.group(1).trim());
                    observedWordSequence.add(m4.group(2).trim());

                } else if (m5.find()) {

                    observedWordSequence.add(m5.group(1).trim());
                    observedWordSequence.add(m5.group(2).trim());
                    observedWordSequence.add(m5.group(3).trim());
                } else if (m6.find()) {

                    observedWordSequence.add(m6.group(1).trim());
                    observedWordSequence.add(m6.group(2).trim());
                } else if (w.contains("@x@name_0") && !w.matches("@x@name_0")) {
                    String realValue = delexicalizedMap.get("@x@name_0");
                    realValue = w.replace("@x@name_0", realValue);
                    delexicalizedMap.put("@x@name_0", realValue);
                    w = "@x@name_0";
                    observedWordSequence.add(w.trim());

                } else if (w.contains("@x@near_0") && !w.equals("@x@near_0")) {
                    String realValue = delexicalizedMap.get("@x@near_0");
                    realValue = w.replace("@x@near_0", realValue);
                    delexicalizedMap.put("@x@near_0", realValue);
                    w = "@x@near_0";
                    observedWordSequence.add(w.trim());
                } else {
                    observedWordSequence.add(w.trim());
                }
            }

            MeaningRepresentation MR = new MeaningRepresentation(singlePredicate, attributeValues, MRPart, delexicalizedMap);

            // We store the maximum observed word sequence length, to use as a limit during generation
            if (observedWordSequence.size() > getMaxWordSequenceLength()) {
                setMaxWordSequenceLength(observedWordSequence.size());
            }
            // We initialize the alignments between words and attribute/value pairs
            ArrayList<String> wordToAttrValueAlignment = new ArrayList<>();
            for (String w : observedWordSequence) {

                if (w.trim().matches("[.,?:;!'\"]")) {
                    wordToAttrValueAlignment.add(Action.TOKEN_PUNCT);
                } else {
                    wordToAttrValueAlignment.add("[]");
                }
            }
            ArrayList<Action> directReferenceSequence = new ArrayList<>();
            for (int r = 0; r < observedWordSequence.size(); r++) {
                directReferenceSequence.add(new Action(observedWordSequence.get(r), wordToAttrValueAlignment.get(r)));
            }
            DatasetInstance DI = new DatasetInstance(MR, directReferenceSequence, postProcessRef(MR, directReferenceSequence));
            getDatasetInstances().get(singlePredicate).stream().filter((existingDI) -> (existingDI.getMeaningRepresentation().getAbstractMR()
                    .equals(DI.getMeaningRepresentation().getAbstractMR()))).map((existingDI) -> {
                existingDI.getEvaluationReferences().addAll(DI.getEvaluationReferences());
                return existingDI;
            }).forEachOrdered((existingDI) -> {
                // We add the direct reference of this DatasetInstance as an available evaluation reference to all previously constructed DatasetInstance that are identical to this one
                DI.getEvaluationReferences().addAll(existingDI.getEvaluationReferences());
            });
            getDatasetInstances().get(singlePredicate).add(DI);
            // value alignments
            HashMap<String, HashMap<String, Double>> observedValueAlignments = new HashMap<>();
            MR.getAttributeValues().keySet().stream().forEach((attr) -> {
                MR.getAttributeValues().get(attr).stream().filter((value) -> (!value.startsWith(Action.TOKEN_X)))
                        .forEachOrdered((value) -> {
                            String valueToCompare = value;
                            //if(valueToCompare.contains("familyfriendly")){
                            //valueToCompare = valueToCompare.replace("familyfriendly", "family friendly");
                            //}
                            observedValueAlignments.put(valueToCompare, new HashMap<String, Double>());
                            // n grams 
                            for (int n = 1; n < observedWordSequence.size(); n++) {
                                //Calculate the similarities between them and valueToCompare
                                for (int r = 0; r <= observedWordSequence.size() - n; r++) {
                                    boolean compareAgainstNGram = true;
                                    for (int j = 0; j < n; j++) {
                                        if (observedWordSequence.get(r + j).startsWith(Action.TOKEN_X)
                                                || wordToAttrValueAlignment.get(r + j).equals(Action.TOKEN_PUNCT)
                                                || observedWordSequence.get(r + j).isEmpty()) {
                                            compareAgainstNGram = false;

                                        }
                                    }
                                    if (compareAgainstNGram) {
                                        String align = "";
                                        String compare = "";
                                        String backwardCompare = "";
                                        for (int j = 0; j < n; j++) {
                                            // The coordinates of the alignment
                                            align += (r + j) + " ";
                                            compare += observedWordSequence.get(r + j);
                                            backwardCompare = observedWordSequence.get(r + j) + backwardCompare;
                                        }
                                        align = align.trim();

                                        // Calculate the character-level distance between the value and the nGram (in its original and reversed order)
                                        Double distance = Levenshtein.getSimilarity(valueToCompare.toLowerCase(), compare.toLowerCase(), true);
                                        Double backwardDistance = Levenshtein.getSimilarity(valueToCompare.toLowerCase(), backwardCompare.toLowerCase(), true);

                                        // We keep the best distance score; note that the Levenshtein distance is normalized so that greater is better 
                                        if (backwardDistance > distance) {
                                            distance = backwardDistance;
                                        }
                                        // We ignore all nGrams that are less similar than a threshold
                                        if (valueToCompare.equals("5 out of 5")
                                                || valueToCompare.equals("1 out of 5")
                                                || valueToCompare.equals("3 out of 5")) {
                                            if (distance > 0.1) {
                                                observedValueAlignments.get(valueToCompare).put(align, distance);
                                            }
                                        } else if (valueToCompare.equals("familyfriendly_no")
                                                || valueToCompare.equals("familyfriendly_yes")) {
                                            if (distance > 0.1) {
                                                observedValueAlignments.get(valueToCompare).put(align, distance);
                                            }

                                        } else if (valueToCompare.equals("more than £30")
                                                || valueToCompare.equals("£20-25")
                                                || valueToCompare.equals("less than £20")) {
                                            if (distance > 0.1) {
                                                observedValueAlignments.get(valueToCompare).put(align, distance);
                                            }
                                        } else {
                                            if (distance > 0.65) {

                                                observedValueAlignments.get(valueToCompare).put(align, distance);

                                            }
                                        }
                                    }

                                }
                            }
                        });
            });
            // We filter out any values that haven't been aligned
            HashSet<String> toRemove = new HashSet<>();
            for (String value : observedValueAlignments.keySet()) {
                if (observedValueAlignments.get(value).isEmpty()) {
                    toRemove.add(value);
                }
            }
            for (String value : toRemove) {
                observedValueAlignments.remove(value);
            }
            while (!observedValueAlignments.keySet().isEmpty()) {
                // Find the best aligned nGram
                Double max = Double.NEGATIVE_INFINITY;
                String[] bestAlignment = new String[2];
                for (String value : observedValueAlignments.keySet()) {
                    for (String alignment : observedValueAlignments.get(value).keySet()) {
                        if (observedValueAlignments.get(value).get(alignment) > max) {
                            max = observedValueAlignments.get(value).get(alignment);
                            bestAlignment[0] = value;
                            bestAlignment[1] = alignment;
                        }
                    }
                }
                // Find the subphrase that corresponds to the best aligned nGram, according to the coordinates
                ArrayList<String> alignedStr = new ArrayList<>();
                String[] coords = bestAlignment[1].split(" ");
                if (coords.length == 1) {
                    alignedStr.add(observedWordSequence.get(Integer.parseInt(coords[0].trim())));
                } else {
                    for (int a = Integer.parseInt(coords[0].trim()); a <= Integer.parseInt(coords[coords.length - 1].trim()); a++) {
                        alignedStr.add(observedWordSequence.get(a));
                    }
                }
                // Store the best aligned nGram
                if (!getValueAlignments().containsKey(bestAlignment[0])) {
                    getValueAlignments().put(bestAlignment[0], new HashMap<ArrayList<String>, Double>());
                }
                getValueAlignments().get(bestAlignment[0]).put(alignedStr, max);
                // And remove it from the observed ones for this instance
                observedValueAlignments.remove(bestAlignment[0]);
                // And also remove any other aligned nGrams that are overlapping with the best aligned nGram
                observedValueAlignments.keySet().forEach((value) -> {
                    HashSet<String> alignmentsToBeRemoved = new HashSet<>();
                    observedValueAlignments.get(value).keySet().forEach((alignment) -> {
                        String[] othCoords = alignment.split(" ");
                        if (Integer.parseInt(coords[0].trim()) <= Integer.parseInt(othCoords[0].trim()) && (Integer.parseInt(coords[coords.length - 1].
                                trim()) >= Integer.parseInt(othCoords[0].trim()))
                                || (Integer.parseInt(othCoords[0].trim()) <= Integer.parseInt(coords[0].trim()) && Integer.parseInt(othCoords[othCoords.length - 1].
                                trim()) >= Integer.parseInt(coords[0].trim()))) {
                            alignmentsToBeRemoved.add(alignment);
                        }
                    });
                    alignmentsToBeRemoved.forEach((alignment) -> {
                        observedValueAlignments.get(value).remove(alignment);
                    });
                });
                // We filter out any values that are no longer aligned (due to overlapping conflicts)
                toRemove = new HashSet<>();
                for (String value : observedValueAlignments.keySet()) {
                    if (observedValueAlignments.get(value).isEmpty()) {
                        toRemove.add(value);
                    }
                }
                for (String value : toRemove) {
                    observedValueAlignments.remove(value);
                }

            }
            getObservedAttrValueSequences().add(observedAttrValueSequence);
        }
    }

    public void writeLists() {
        String file1 = "cache/getPredicates()";
        String file2 = "cache/attributes";
        String file3 = "cache/attributeValuePairs";
        String file4 = "cache/getValueAlignments()";
        String file5 = "cache/getDatasetInstances";
        String file6 = "cache/maxLengths";
        String file7 = "cache/getValidationDatasetInstances";
        String file8 = "cache/getTrainingDatasetInstances";
        FileOutputStream fout1 = null;
        ObjectOutputStream oos1 = null;
        FileOutputStream fout2 = null;
        ObjectOutputStream oos2 = null;
        FileOutputStream fout3 = null;
        ObjectOutputStream oos3 = null;
        FileOutputStream fout4 = null;
        ObjectOutputStream oos4 = null;
        FileOutputStream fout5 = null;
        ObjectOutputStream oos5 = null;
        FileOutputStream fout6 = null;
        ObjectOutputStream oos6 = null;
        FileOutputStream fout7 = null;
        ObjectOutputStream oos7 = null;
        FileOutputStream fout8 = null;
        ObjectOutputStream oos8 = null;
        try {
            System.out.println("Write lists...");
            fout1 = new FileOutputStream(file1);
            oos1 = new ObjectOutputStream(fout1);
            oos1.writeObject(getPredicates());
            ///////////////////
            fout2 = new FileOutputStream(file2);
            oos2 = new ObjectOutputStream(fout2);
            oos2.writeObject(getAttributes());
            ///////////////////
            fout3 = new FileOutputStream(file3);
            oos3 = new ObjectOutputStream(fout3);
            oos3.writeObject(getAttributeValuePairs());
            ///////////////////
            fout4 = new FileOutputStream(file4);
            oos4 = new ObjectOutputStream(fout4);
            oos4.writeObject(getValueAlignments());
            ///////////////////
            fout5 = new FileOutputStream(file5);
            oos5 = new ObjectOutputStream(fout5);
            oos5.writeObject(getDatasetInstances());
            ///////////////////
            fout6 = new FileOutputStream(file6);
            oos6 = new ObjectOutputStream(fout6);
            //ArrayList<Integer> lengths = new ArrayList<Integer>();
            //lengths.add(getMaxContentSequenceLength());
            //lengths.add(getMaxWordSequenceLength());
            oos6.writeObject(getMaxWordSequenceLength());
            fout7 = new FileOutputStream(file7);
            oos7 = new ObjectOutputStream(fout7);
            oos7.writeObject(getValidationData());
            fout8 = new FileOutputStream(file8);
            oos8 = new ObjectOutputStream(fout8);
            oos8.writeObject(getTrainingData());
        } catch (IOException ex) {
        } finally {
            try {
                fout1.close();
                fout2.close();
                fout3.close();
                fout4.close();
                fout5.close();
                fout6.close();
            } catch (IOException ex) {
            }
            try {
                oos1.close();
                oos2.close();
                oos3.close();
                oos4.close();
                oos5.close();
                oos6.close();
            } catch (IOException ex) {
            }
        }
    }

    @SuppressWarnings("unchecked")
    public boolean loadLists() {
        String file1 = "cache/getPredicates()";
        String file2 = "cache/attributes";
        String file3 = "cache/attributeValuePairs";
        String file4 = "cache/getValueAlignments()";
        String file5 = "cache/getDatasetInstances";
        String file6 = "cache/maxLengths";
        String file7 = "cache/getValidationDatasetInstances";
        String file8 = "cache/getTrainingDatasetInstances";
        FileInputStream fin1 = null;
        ObjectInputStream ois1 = null;
        FileInputStream fin2 = null;
        ObjectInputStream ois2 = null;
        FileInputStream fin3 = null;
        ObjectInputStream ois3 = null;
        FileInputStream fin4 = null;
        ObjectInputStream ois4 = null;
        FileInputStream fin5 = null;
        ObjectInputStream ois5 = null;
        FileInputStream fin6 = null;
        ObjectInputStream ois6 = null;
        FileInputStream fin7 = null;
        ObjectInputStream ois7 = null;
        FileInputStream fin8 = null;
        ObjectInputStream ois8 = null;
        if ((new File(file1)).exists()
                && (new File(file2)).exists()
                && (new File(file3)).exists()
                && (new File(file4)).exists()
                && (new File(file5)).exists()
                && (new File(file6)).exists()
                && (new File(file7)).exists()
                && (new File(file8)).exists()) {
            try {
                System.out.println("Load lists...");
                fin1 = new FileInputStream(file1);
                ois1 = new ObjectInputStream(fin1);
                Object o1 = ois1.readObject();
                if (getPredicates() == null) {
                    if (o1 instanceof ArrayList) {
                        setPredicates(new ArrayList<String>((Collection<? extends String>) o1));
                    }
                } else if (o1 instanceof ArrayList) {
                    getPredicates().addAll((Collection<? extends String>) o1);
                }
                ///////////////////
                fin2 = new FileInputStream(file2);
                ois2 = new ObjectInputStream(fin2);
                Object o2 = ois2.readObject();
                if (getAttributes() == null) {
                    if (o2 instanceof HashMap) {
                        setAttributes(new HashMap<String, HashSet<String>>((Map<? extends String, ? extends HashSet<String>>) o2));
                    }
                } else if (o2 instanceof HashMap) {
                    getAttributes().putAll((Map<? extends String, ? extends HashSet<String>>) o2);
                }
                ///////////////////
                fin3 = new FileInputStream(file3);
                ois3 = new ObjectInputStream(fin3);
                Object o3 = ois3.readObject();
                if (getAttributeValuePairs() == null) {
                    if (o3 instanceof HashMap) {
                        setAttributeValuePairs(new HashMap<String, HashSet<String>>((Map<? extends String, ? extends HashSet<String>>) o3));
                    }
                } else if (o3 instanceof HashMap) {
                    getAttributeValuePairs().putAll((Map<? extends String, ? extends HashSet<String>>) o3);
                }
                ///////////////////
                fin4 = new FileInputStream(file4);
                ois4 = new ObjectInputStream(fin4);
                Object o4 = ois4.readObject();
                if (getValueAlignments() == null) {
                    if (o4 instanceof HashMap) {
                        setValueAlignments(new HashMap<String, HashMap<ArrayList<String>, Double>>((Map<? extends String, ? extends HashMap<ArrayList<String>, Double>>) o4));
                    }
                } else if (o4 instanceof HashMap) {
                    getValueAlignments().putAll((Map<? extends String, ? extends HashMap<ArrayList<String>, Double>>) o4);
                }
                ///////////////////
                fin5 = new FileInputStream(file5);
                ois5 = new ObjectInputStream(fin5);
                Object o5 = ois5.readObject();
                if (getDatasetInstances() == null) {
                    if (o5 instanceof HashMap) {
                        setDatasetInstances(new HashMap<String, ArrayList<DatasetInstance>>((Map<? extends String, ? extends ArrayList<DatasetInstance>>) o5));
                    }
                } else if (o5 instanceof HashMap) {
                    getDatasetInstances().putAll((Map<? extends String, ? extends ArrayList<DatasetInstance>>) o5);
                }
                ///////////////////
                fin6 = new FileInputStream(file6);
                ois6 = new ObjectInputStream(fin6);
                Object o6 = ois6.readObject();
                //ArrayList<Integer> lengths = new ArrayList<Integer>((Collection<? extends Integer>) o6);
                //setMaxContentSequenceLength(lengths.get(0));
                setMaxWordSequenceLength((Integer) o6);
                ///////////////////
                fin7 = new FileInputStream(file7);
                ois7 = new ObjectInputStream(fin7);
                Object o7 = ois7.readObject();
                if (getValidationData() == null) {
                    if (o7 instanceof ArrayList) {
                        setValidationData(new ArrayList<DatasetInstance>((Collection<? extends DatasetInstance>) o7));
                    }
                } else if (o7 instanceof ArrayList) {
                    getValidationData().addAll(new ArrayList<DatasetInstance>((Collection<? extends DatasetInstance>) o7));
                }
                fin8 = new FileInputStream(file8);
                ois8 = new ObjectInputStream(fin8);
                Object o8 = ois8.readObject();
                if (getTrainingData() == null) {
                    if (o8 instanceof ArrayList) {
                        setTrainingData(new ArrayList<DatasetInstance>((Collection<? extends DatasetInstance>) o8));
                    }
                } else if (o8 instanceof ArrayList) {
                    getTrainingData().addAll(new ArrayList<DatasetInstance>((Collection<? extends DatasetInstance>) o8));
                }

                System.out.println("done!");
            } catch (ClassNotFoundException | IOException ex) {
            } finally {
                try {
                    fin1.close();
                    fin2.close();
                    fin3.close();
                    fin4.close();
                    fin5.close();
                    fin6.close();
                    fin7.close();
                    fin8.close();
                } catch (IOException ex) {
                }
                try {
                    ois1.close();
                    ois2.close();
                    ois3.close();
                    ois4.close();
                    ois5.close();
                    ois6.close();
                    ois7.close();
                    ois8.close();
                } catch (IOException ex) {
                }
            }
            return true;
        } else {
            return false;
        }
    }

    /**
     *
     * @return
     */
    public boolean loadAvailableActions() {
        if (!isCache()) {
            return false;
        }
        String file1 = "cache/availableContentActions_SF_" + getDataset();
        String file2 = "cache/availableWordActions_SF_" + getDataset();
        FileInputStream fin1 = null;
        ObjectInputStream ois1 = null;
        FileInputStream fin2 = null;
        ObjectInputStream ois2 = null;
        if ((new File(file1)).exists()
                && (new File(file2)).exists()) {
            try {
                System.out.print("Load available actions...");

                fin1 = new FileInputStream(file1);
                ois1 = new ObjectInputStream(fin1);
                Object o1 = ois1.readObject();
                if (getAvailableContentActions() == null) {
                    if (o1 instanceof HashMap) {
                        setAvailableContentActions((HashMap<String, HashSet<String>>) o1);
                    }
                } else if (o1 instanceof HashMap) {
                    getAvailableContentActions().putAll((HashMap<String, HashSet<String>>) o1);
                }

                fin2 = new FileInputStream(file2);
                ois2 = new ObjectInputStream(fin2);
                Object o2 = ois2.readObject();
                if (getAvailableWordActions() == null) {
                    if (o2 instanceof HashMap) {
                        setAvailableWordActions((HashMap<String, HashMap<String, HashSet<Action>>>) o2);
                    }
                } else if (o2 instanceof HashMap) {
                    getAvailableWordActions().putAll((HashMap<String, HashMap<String, HashSet<Action>>>) o2);
                }
                System.out.println("done!");
            } catch (ClassNotFoundException | IOException ex) {
            } finally {
                try {
                    fin1.close();
                    fin2.close();
                } catch (IOException ex) {
                }
                try {
                    ois1.close();
                    ois2.close();
                } catch (IOException ex) {
                }
            }
        } else {
            return false;
        }
        return true;
    }

    /**
     *
     */
    public void writeAvailableActions() {
        String file1 = "cache/availableContentActions_SF_" + getDataset();
        String file2 = "cache/availableWordActions_SF_" + getDataset();
        FileOutputStream fout1 = null;
        ObjectOutputStream oos1 = null;
        FileOutputStream fout2 = null;
        ObjectOutputStream oos2 = null;
        try {
            System.out.print("Write available actions...");
            fout1 = new FileOutputStream(file1);
            oos1 = new ObjectOutputStream(fout1);
            oos1.writeObject(getAvailableContentActions());

            fout2 = new FileOutputStream(file2);
            oos2 = new ObjectOutputStream(fout2);
            oos2.writeObject(getAvailableWordActions());
            System.out.println("done!");
        } catch (IOException ex) {
        } finally {
            try {
                fout1.close();
                fout2.close();
            } catch (IOException ex) {
            }
            try {
                oos1.close();
                oos2.close();
            } catch (IOException ex) {
            }
        }
    }

    @Override
    public void createTrainingData() {

        int N = 200;
        System.out.println("create naive alignments");
        createNaiveAlignments(getTrainingData());
        writeObservedAttrValues();
        System.out.println("done");
        ArrayList<String> topNWords = new ArrayList<>();
        // build word dictionary
        HashMap<String, Integer> wordDictionary = new HashMap<>();
        getTrainingData().forEach((di) -> {
            di.getDirectReferenceSequence().forEach((action) -> {
                if (!wordDictionary.containsKey(action.getWord())) {
                    wordDictionary.put(action.getWord(), 1);
                }
                wordDictionary.put(action.getWord(), wordDictionary.get(action.getWord()) + 1);
            });

        });
        /*
		ValueComparator bvc = new ValueComparator(wordDictionary);
		
        TreeMap<String, Integer> sorted_map = new TreeMap<String, Integer>(bvc);
        sorted_map.putAll(wordDictionary);
        for(String word: sorted_map.keySet()){
        	if(topNWords.size()<N){
        		topNWords.add(word);
        	}
        	else{
        		break;
        	}
        }
        
        getTrainingData().forEach((di)->{
        	for(Action a : di.getDirectReferenceSequence()){
        		if(!topNWords.contains(a.getWord())){
        			a.setWord("@unk@");
        		}
        	}
        	
        });*/

        // Create (or load from cache) the content and word language models per predicate
        // These are LMs trained on full LMs, used for determining how @unk@ tokens should be realized
        if (isResetStoredCaches() || !loadLMs()) {
            HashMap<String, ArrayList<ArrayList<String>>> LMWordTrainingPerPred = new HashMap<>();
            HashMap<String, ArrayList<ArrayList<String>>> LMAttrTrainingPerPred = new HashMap<>();
            getTrainingData().stream().map((di) -> {
                if (!LMWordTrainingPerPred.containsKey(di.getMeaningRepresentation().getPredicate())) {
                    LMWordTrainingPerPred.put(di.getMeaningRepresentation().getPredicate(), new ArrayList<ArrayList<String>>());
                    LMAttrTrainingPerPred.put(di.getMeaningRepresentation().getPredicate(), new ArrayList<ArrayList<String>>());
                }
                return di;
            }).forEachOrdered((di) -> {
                HashSet<ArrayList<Action>> seqs = new HashSet<>();
                seqs.add(di.getDirectReferenceSequence());
                seqs.forEach((seq) -> {
                    ArrayList<String> wordSeq = new ArrayList<>();
                    ArrayList<String> attrSeq = new ArrayList<>();

                    // We add some empty tokens at the start of each sequence
                    wordSeq.add("@@");
                    wordSeq.add("@@");
                    attrSeq.add("@@");
                    attrSeq.add("@@");
                    for (int i = 0; i < seq.size(); i++) {
                        if (!seq.get(i).getAttribute().equals(Action.TOKEN_END)
                                && !seq.get(i).getWord().equals(Action.TOKEN_END)) {
                            wordSeq.add(seq.get(i).getWord());
                        }
                        if (attrSeq.isEmpty()) {
                            attrSeq.add(seq.get(i).getAttribute());
                        } else if (!attrSeq.get(attrSeq.size() - 1).equals(seq.get(i).getAttribute())) {
                            attrSeq.add(seq.get(i).getAttribute());
                        }
                    }
                    wordSeq.add(Action.TOKEN_END);
                    LMWordTrainingPerPred.get(di.getMeaningRepresentation().getPredicate()).add(wordSeq);
                    LMAttrTrainingPerPred.get(di.getMeaningRepresentation().getPredicate()).add(attrSeq);
                });
            });

            setWordFullLMsPerPredicate(new HashMap<>());
            setContentFullLMsPerPredicate(new HashMap<>());
            LMWordTrainingPerPred.keySet().stream().map((pred) -> {
                SimpleLM simpleWordLM = new SimpleLM(3);
                simpleWordLM.trainOnStrings(LMWordTrainingPerPred.get(pred));
                getWordFullLMsPerPredicate().put(pred, simpleWordLM);
                return pred;
            }).forEachOrdered((pred) -> {
                SimpleLM simpleAttrLM = new SimpleLM(3);
                simpleAttrLM.trainOnStrings(LMAttrTrainingPerPred.get(pred));
                getContentFullLMsPerPredicate().put(pred, simpleAttrLM);
            });
        }

        // Go through the sequences in the data and populate the available content and word action dictionaries
        // We populate a distinct word dictionary for each attribute, and populate it with the words of word sequences whose corresponding content sequences contain that attribute
        HashMap<String, HashSet<String>> availableContentActions = new HashMap<>();
        HashMap<String, HashMap<String, HashSet<Action>>> availableWordActions = new HashMap<>();
        HashMap<String, HashMap<String, HashMap<Action, Integer>>> availableWordActionCounts = new HashMap<>();
        availableContentActions.put(singlePredicate, new HashSet<>());
        availableWordActions.put(singlePredicate, new HashMap<>());
        availableWordActionCounts.put(singlePredicate, new HashMap<>());
        getTrainingData().forEach((DI) -> {
            ArrayList<Action> realization = DI.getDirectReferenceSequence();
            realization.stream().filter((a) -> (!a.getAttribute().equals(Action.TOKEN_END))).forEachOrdered((Action a) -> {
                String attr;
                if (a.getAttribute().contains("=")) {
                    attr = a.getAttribute().substring(0, a.getAttribute().indexOf('='));
                } else {
                    attr = a.getAttribute();
                }
                if (!attr.trim().isEmpty()) {
                    availableContentActions.get(singlePredicate).add(attr);
                }
                if (!availableWordActions.get(singlePredicate).containsKey(attr)) {
                    availableWordActions.get(singlePredicate).put(attr, new HashSet<Action>());
                    availableWordActions.get(singlePredicate).get(attr).add(new Action(Action.TOKEN_END, attr));
                    availableWordActionCounts.get(singlePredicate).put(attr, new HashMap<Action, Integer>());
                }
                if (!a.getWord().equals(Action.TOKEN_START)
                        && !a.getWord().equals(Action.TOKEN_END)
                        && !a.getWord().matches("([,.?!;:'])")) {
                    if (a.getWord().startsWith(Action.TOKEN_X)) {
                        if (a.getWord().substring(3, a.getWord().lastIndexOf('_')).toLowerCase().trim().equals(attr)) {
                            if (!a.getWord().trim().isEmpty()) {
                                availableWordActions.get(singlePredicate).get(attr).add(new Action(a.getWord(), attr));
                            }
                        }
                    } else {
                        if (!a.getWord().trim().isEmpty()) {
                            Action act = new Action(a.getWord(), attr);
                            if (!availableWordActionCounts.get(singlePredicate).get(attr).containsKey(act)) {
                                availableWordActionCounts.get(singlePredicate).get(attr).put(act, 1);
                            } else {
                                availableWordActionCounts.get(singlePredicate).get(attr).put(act, availableWordActionCounts.get(singlePredicate).get(attr).get(act) + 1);
                            }
                        }
                    }
                }
            });
        });
        // Do not learn weights for all actions but only those that are frequent enough
        int actionThreshold = 5;
        System.out.println("TRIM ACTIONS");
        for (String predicate : availableWordActionCounts.keySet()) {
            for (String attr : availableWordActionCounts.get(predicate).keySet()) {
                //System.out.println("+++ " + attr + " +++");
                for (Action act : availableWordActionCounts.get(predicate).get(attr).keySet()) {
                    //System.out.println("\t" + act + ": " + availableWordActionCounts.get(predicate).get(attr).get(act));
                    if (availableWordActionCounts.get(predicate).get(attr).get(act) > actionThreshold) {
                        availableWordActions.get(predicate).get(attr).add(act);
                    }
                }
                System.out.println("+++ " + attr + " +++ from " + availableWordActionCounts.get(predicate).get(attr).keySet().size() + " to " + availableWordActions.get(predicate).get(attr).size());
            }
        }
        /*for (String predicate : availableWordActions.keySet()) {
            for (String attr : availableWordActions.get(predicate).keySet()) {
                System.out.println("+++ " + attr + " +++");
                for (Action act : availableWordActionCounts.get(predicate).get(attr).keySet()) {
                    System.out.println("\t" + act);
                }
            }
        }*/
        setAvailableContentActions(availableContentActions);
        setAvailableWordActions(availableWordActions);
        writeAvailableActions();
        // Replace infrequent actions with @unk@ in the sequences
        getTrainingData().forEach((DI) -> {
            String predicate = DI.getMeaningRepresentation().getPredicate();
            for (int r = 0; r < DI.getDirectReferenceSequence().size(); r++) {
                if (!DI.getDirectReferenceSequence().get(r).getAttribute().equals(Action.TOKEN_END)) {
                    String attr;
                    if (DI.getDirectReferenceSequence().get(r).getAttribute().contains("=")) {
                        attr = DI.getDirectReferenceSequence().get(r).getAttribute().substring(0, DI.getDirectReferenceSequence().get(r).getAttribute().indexOf('='));
                    } else {
                        attr = DI.getDirectReferenceSequence().get(r).getAttribute();
                    }
                    if (!DI.getDirectReferenceSequence().get(r).getWord().equals(Action.TOKEN_START)
                            && !DI.getDirectReferenceSequence().get(r).getWord().equals(Action.TOKEN_END)
                            && !DI.getDirectReferenceSequence().get(r).getWord().matches("([,.?!;:'])")) {
                        if (!DI.getDirectReferenceSequence().get(r).getWord().startsWith(Action.TOKEN_X)) {
                            if (!availableWordActions.get(predicate).get(attr).contains(DI.getDirectReferenceSequence().get(r))) {
                                DI.getDirectReferenceSequence().get(r).setWord("@unk@");
                            }
                        }
                    }
                }
            }
        });
        // Create LMs for the sequences where infrequent words have been replaced with @unk@
        if (isResetStoredCaches() || !loadLMs()) {
            HashMap<String, ArrayList<ArrayList<String>>> LMWordTrainingPerPred = new HashMap<>();
            HashMap<String, ArrayList<ArrayList<String>>> LMAttrTrainingPerPred = new HashMap<>();
            getTrainingData().stream().map((di) -> {
                if (!LMWordTrainingPerPred.containsKey(di.getMeaningRepresentation().getPredicate())) {
                    LMWordTrainingPerPred.put(di.getMeaningRepresentation().getPredicate(), new ArrayList<ArrayList<String>>());
                    LMAttrTrainingPerPred.put(di.getMeaningRepresentation().getPredicate(), new ArrayList<ArrayList<String>>());
                }
                return di;
            }).forEachOrdered((di) -> {
                HashSet<ArrayList<Action>> seqs = new HashSet<>();
                seqs.add(di.getDirectReferenceSequence());
                seqs.forEach((seq) -> {
                    ArrayList<String> wordSeq = new ArrayList<>();
                    ArrayList<String> attrSeq = new ArrayList<>();

                    // We add some empty tokens at the start of each sequence
                    wordSeq.add("@@");
                    wordSeq.add("@@");
                    attrSeq.add("@@");
                    attrSeq.add("@@");
                    for (int i = 0; i < seq.size(); i++) {
                        if (!seq.get(i).getAttribute().equals(Action.TOKEN_END)
                                && !seq.get(i).getWord().equals(Action.TOKEN_END)) {
                            wordSeq.add(seq.get(i).getWord());
                        }
                        if (attrSeq.isEmpty()) {
                            attrSeq.add(seq.get(i).getAttribute());
                        } else if (!attrSeq.get(attrSeq.size() - 1).equals(seq.get(i).getAttribute())) {
                            attrSeq.add(seq.get(i).getAttribute());
                        }
                    }
                    wordSeq.add(Action.TOKEN_END);
                    LMWordTrainingPerPred.get(di.getMeaningRepresentation().getPredicate()).add(wordSeq);
                    LMAttrTrainingPerPred.get(di.getMeaningRepresentation().getPredicate()).add(attrSeq);
                });
            });

            setWordLMsPerPredicate(new HashMap<>());
            setContentLMsPerPredicate(new HashMap<>());
            LMWordTrainingPerPred.keySet().stream().map((pred) -> {
                SimpleLM simpleWordLM = new SimpleLM(3);
                simpleWordLM.trainOnStrings(LMWordTrainingPerPred.get(pred));
                getWordLMsPerPredicate().put(pred, simpleWordLM);
                return pred;
            }).forEachOrdered((pred) -> {
                SimpleLM simpleAttrLM = new SimpleLM(3);
                simpleAttrLM.trainOnStrings(LMAttrTrainingPerPred.get(pred));
                getContentLMsPerPredicate().put(pred, simpleAttrLM);
            });
            writeLMs();
        }

        //getAvailableWordActions().get(singlePredicate).keySet().forEach((attr)->{
        //System.out.println(attr+" "+getAvailableWordActions().get(singlePredicate).get(attr).size());
        //});
        // create training instances
        //if(isResetStoredCaches()||!loadTrainingData(getTrainingData().size())){
        /*System.out.print("Create training data...");
        Object[] results = inferFeatureAndCostVectors();
        System.out.print("almost...");
        @SuppressWarnings("unchecked")
        ConcurrentHashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>> getPredicateContentTrainingDataBefore = (ConcurrentHashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>) results[0];
        @SuppressWarnings("unchecked")
        ConcurrentHashMap<DatasetInstance, HashMap<String, HashMap<String, ArrayList<Instance>>>> getPredicateWordTrainingDataBefore = (ConcurrentHashMap<DatasetInstance, HashMap<String, HashMap<String, ArrayList<Instance>>>>) results[1];
        // Reorganize the feature/cost vector collections 
        // Initially they are mapped according to DatasetInstance (since it helps with parallel processing) but we prefer them mapped by predicate for training
        setPredicateContentTrainingData(new HashMap<>());
        getTrainingData().forEach((di) -> {
            getPredicateContentTrainingDataBefore.get(di).keySet().stream().map((predicate) -> {
                if (!getPredicateContentTrainingData().containsKey(predicate)) {
                    getPredicateContentTrainingData().put(predicate, new ArrayList<Instance>());
                }
                return predicate;
            }).forEachOrdered((predicate) -> {
                getPredicateContentTrainingData().get(predicate).addAll(getPredicateContentTrainingDataBefore.get(di).get(predicate));
            });
        });
        setPredicateWordTrainingData(new HashMap<>());
        getTrainingData().forEach((di) -> {
            getPredicateWordTrainingDataBefore.get(di).keySet().stream().map((predicate) -> {
                if (!getPredicateWordTrainingData().containsKey(predicate)) {
                    getPredicateWordTrainingData().put(predicate, new HashMap<String, ArrayList<Instance>>());
                }
                return predicate;
            }).forEachOrdered((predicate) -> {
                getPredicateWordTrainingDataBefore.get(di).get(predicate).keySet().stream().map((attribute) -> {
                    if (!getPredicateWordTrainingData().get(predicate).containsKey(attribute)) {
                        getPredicateWordTrainingData().get(predicate).put(attribute, new ArrayList<Instance>());
                    }
                    return attribute;
                }).forEachOrdered((attribute) -> {
                    getPredicateWordTrainingData().get(predicate).get(attribute).addAll(getPredicateWordTrainingDataBefore.get(di).get(predicate).get(attribute));
                });
            });
        });
        writeTrainingData(getTrainingData().size());*/
        //}
        if (isResetStoredCaches() || !loadInitClassifiers(getTrainingData().size(), ILEngine.trainedAttrClassifiers_0, ILEngine.trainedWordClassifiers_0)) {
            List<List<DatasetInstance>> trainingDataBatches = Lists.partition(getTrainingData(), JLOLS.batchSize);

            int b = 0;
            for (List<DatasetInstance> batch : trainingDataBatches) {
                System.out.print("Infering vectors from batch " + b + " out of " + trainingDataBatches.size());
                Object[] results = inferFeatureAndCostVectors(batch);

                @SuppressWarnings("unchecked")
                ConcurrentHashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>> getPredicateContentTrainingDataBefore = (ConcurrentHashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>>) results[0];
                @SuppressWarnings("unchecked")
                ConcurrentHashMap<DatasetInstance, HashMap<String, HashMap<String, ArrayList<Instance>>>> getPredicateWordTrainingDataBefore = (ConcurrentHashMap<DatasetInstance, HashMap<String, HashMap<String, ArrayList<Instance>>>>) results[1];

                // Reorganize the feature/cost vector collections 
                // Initially they are mapped according to DatasetInstance (since it helps with parallel processing) but we prefer them mapped by predicate for training
                setPredicateContentTrainingData(new HashMap<>());
                setPredicateWordTrainingData(new HashMap<>());

                b++;
                batch.forEach((di) -> {
                    getPredicateContentTrainingDataBefore.get(di).keySet().stream().map((predicate) -> {
                        if (!getPredicateContentTrainingData().containsKey(predicate)) {
                            getPredicateContentTrainingData().put(predicate, new ArrayList<Instance>());
                        }
                        return predicate;
                    }).forEachOrdered((predicate) -> {
                        getPredicateContentTrainingData().get(predicate).addAll(getPredicateContentTrainingDataBefore.get(di).get(predicate));
                    });
                });
                batch.forEach((di) -> {
                    getPredicateWordTrainingDataBefore.get(di).keySet().stream().map((predicate) -> {
                        if (!getPredicateWordTrainingData().containsKey(predicate)) {
                            getPredicateWordTrainingData().put(predicate, new HashMap<String, ArrayList<Instance>>());
                        }
                        return predicate;
                    }).forEachOrdered((predicate) -> {
                        getPredicateWordTrainingDataBefore.get(di).get(predicate).keySet().stream().map((attribute) -> {
                            if (!getPredicateWordTrainingData().get(predicate).containsKey(attribute)) {
                                getPredicateWordTrainingData().get(predicate).put(attribute, new ArrayList<Instance>());
                            }
                            return attribute;
                        }).forEachOrdered((attribute) -> {
                            getPredicateWordTrainingData().get(predicate).get(attribute).addAll(getPredicateWordTrainingDataBefore.get(di).get(predicate).get(attribute));
                        });
                    });
                });
                System.out.println(" done!");
                ILEngine.runInitialTrainingOnBatch();

                //Clear already used vectors
                for (String predicate : getPredicateContentTrainingData().keySet()) {
                    getPredicateContentTrainingData().get(predicate).clear();
                }
                for (String predicate : getPredicateWordTrainingData().keySet()) {
                    for (String attribute : getPredicateWordTrainingData().get(predicate).keySet()) {
                        getPredicateWordTrainingData().get(predicate).get(attribute).clear();
                    }
                }
            }
            ILEngine.writeInitClassifiers();
        }
    }

    @SuppressWarnings("unchecked")
    public boolean loadTrainingData(int dataSize) {
        String file1 = "cache/attrTrainingData" + "_" + dataSize;
        String file2 = "cache/wordTrainingData" + "_" + dataSize;
        FileInputStream fin1 = null;
        ObjectInputStream ois1 = null;
        FileInputStream fin2 = null;
        ObjectInputStream ois2 = null;
        if ((new File(file1)).exists()
                && (new File(file2)).exists()) {
            try {
                System.out.println("Load training data...");
                fin1 = new FileInputStream(file1);
                ois1 = new ObjectInputStream(fin1);
                Object o1 = ois1.readObject();
                if (getPredicateContentTrainingData() == null) {
                    if (o1 instanceof HashMap) {
                        setPredicateContentTrainingData(new HashMap<String, ArrayList<Instance>>((Map<? extends String, ? extends ArrayList<Instance>>) o1));
                    }
                } else if (o1 instanceof HashMap) {
                    getPredicateContentTrainingData().putAll((Map<? extends String, ? extends ArrayList<Instance>>) o1);
                }

                fin2 = new FileInputStream(file2);
                ois2 = new ObjectInputStream(fin2);
                Object o2 = ois2.readObject();
                if (getPredicateWordTrainingData() == null) {
                    if (o2 instanceof HashMap) {
                        setPredicateWordTrainingData(new HashMap<String, HashMap<String, ArrayList<Instance>>>((Map<? extends String, ? extends HashMap<String, ArrayList<Instance>>>) o2));
                    }
                } else if (o2 instanceof HashMap) {
                    getPredicateWordTrainingData().putAll((Map<? extends String, ? extends HashMap<String, ArrayList<Instance>>>) o2);
                }

            } catch (ClassNotFoundException | IOException ex) {
            } finally {
                try {
                    fin1.close();
                    fin2.close();
                } catch (IOException ex) {
                }
                try {
                    ois1.close();
                    ois2.close();
                } catch (IOException ex) {
                }
            }
        } else {
            return false;
        }
        return true;
    }

    public void writeTrainingData(int dataSize) {
        String file1 = "cache/attrTrainingData" + "_" + dataSize;
        String file2 = "cache/wordTrainingData" + "_" + dataSize;
        FileOutputStream fout1 = null;
        ObjectOutputStream oos1 = null;
        FileOutputStream fout2 = null;
        ObjectOutputStream oos2 = null;
        try {
            System.out.print("Write Training Data...");
            fout1 = new FileOutputStream(file1);
            oos1 = new ObjectOutputStream(fout1);
            oos1.writeObject(getPredicateContentTrainingData());

            fout2 = new FileOutputStream(file2);
            oos2 = new ObjectOutputStream(fout2);
            oos2.writeObject(getPredicateWordTrainingData());

        } catch (IOException ex) {
        } finally {
            try {
                fout1.close();
                fout2.close();
            } catch (IOException ex) {
            }
            try {
                oos1.close();
                oos2.close();
            } catch (IOException ex) {
            }
        }
    }

    public Object[] inferFeatureAndCostVectors(List<DatasetInstance> trainingData) {
        ConcurrentHashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>> contentTrainingData = new ConcurrentHashMap<>();
        ConcurrentHashMap<DatasetInstance, HashMap<String, HashMap<String, ArrayList<Instance>>>> wordTrainingData = new ConcurrentHashMap<>();
        if (!getAvailableWordActions().isEmpty() && !getPredicates().isEmpty()) {
            // Initialize collections
            trainingData.stream().map((di) -> {
                contentTrainingData.put(di, new HashMap<String, ArrayList<Instance>>());
                return di;
            }).map((di) -> {
                wordTrainingData.put(di, new HashMap<String, HashMap<String, ArrayList<Instance>>>());
                return di;
            }).forEachOrdered((di) -> {
                getPredicates().stream().map((predicate) -> {
                    contentTrainingData.get(di).put(predicate, new ArrayList<Instance>());
                    return predicate;
                }).map((predicate) -> {
                    wordTrainingData.get(di).put(predicate, new HashMap<String, ArrayList<Instance>>());
                    return predicate;
                }).forEachOrdered((predicate) -> {
                    getAttributes().get(predicate).stream().filter((attribute) -> (!wordTrainingData.get(di).get(predicate).containsKey(attribute))).forEachOrdered((attribute) -> {
                        wordTrainingData.get(di).get(predicate).put(attribute, new ArrayList<Instance>());
                    });
                });
            });
            // Infer the vectors in parallel processes to save time
            ExecutorService executor = Executors.newFixedThreadPool(THREAD_COUNT);
            //ArrayList<DatasetInstance> newList = new ArrayList<>();
            //for(int w =0;w<100;w++){
            //newList.add(getTrainingData().get(w));
            //}

            trainingData.forEach((di) -> {
                //newList.forEach((di)->{

                executor.execute(new InferE2EVectorsThread(di, this, contentTrainingData, wordTrainingData));
            });
            executor.shutdown();
            while (!executor.isTerminated()) {
            }
        }

        Object[] results = new Object[2];
        results[0] = contentTrainingData;
        results[1] = wordTrainingData;
        return results;
    }

    /**
     *
     * @return
     */
    public boolean loadLMs() {
        if (!isCache()) {
            return false;
        }
        String file2 = "cache/wordLMs_SF_" + getDataset();
        String file3 = "cache/attrLMs_SF_" + getDataset();
        String file4 = "cache/wordFullLMs_SF_" + getDataset();
        String file5 = "cache/attrFullLMs_SF_" + getDataset();
        FileInputStream fin2 = null;
        ObjectInputStream ois2 = null;
        FileInputStream fin3 = null;
        ObjectInputStream ois3 = null;
        FileInputStream fin4 = null;
        ObjectInputStream ois4 = null;
        FileInputStream fin5 = null;
        ObjectInputStream ois5 = null;
        if ((new File(file2)).exists()
                && (new File(file3)).exists()
                && (new File(file4)).exists()
                && (new File(file5)).exists()) {
            try {
                System.out.print("Load language models...");

                fin2 = new FileInputStream(file2);
                ois2 = new ObjectInputStream(fin2);
                Object o2 = ois2.readObject();
                if (getWordLMsPerPredicate() == null) {
                    if (o2 instanceof HashMap) {
                        setWordLMsPerPredicate(new HashMap<String, SimpleLM>((Map<? extends String, ? extends SimpleLM>) o2));
                    }
                } else if (o2 instanceof HashMap) {
                    getWordLMsPerPredicate().putAll((Map<? extends String, ? extends SimpleLM>) o2);
                }

                fin3 = new FileInputStream(file3);
                ois3 = new ObjectInputStream(fin3);
                Object o3 = ois3.readObject();
                if (getContentLMsPerPredicate() == null) {
                    if (o3 instanceof HashMap) {
                        setContentLMsPerPredicate(new HashMap<String, SimpleLM>((Map<? extends String, ? extends SimpleLM>) o3));
                    }
                } else if (o3 instanceof HashMap) {
                    getContentLMsPerPredicate().putAll((Map<? extends String, ? extends SimpleLM>) o3);
                }

                fin4 = new FileInputStream(file4);
                ois4 = new ObjectInputStream(fin4);
                Object o4 = ois4.readObject();
                if (getContentLMsPerPredicate() == null) {
                    if (o4 instanceof HashMap) {
                        setWordFullLMsPerPredicate(new HashMap<String, SimpleLM>((Map<? extends String, ? extends SimpleLM>) o4));
                    }
                } else if (o4 instanceof HashMap) {
                    getWordFullLMsPerPredicate().putAll((Map<? extends String, ? extends SimpleLM>) o4);
                }

                fin5 = new FileInputStream(file5);
                ois5 = new ObjectInputStream(fin5);
                Object o5 = ois5.readObject();
                if (getContentLMsPerPredicate() == null) {
                    if (o5 instanceof HashMap) {
                        setContentFullLMsPerPredicate(new HashMap<String, SimpleLM>((Map<? extends String, ? extends SimpleLM>) o5));
                    }
                } else if (o5 instanceof HashMap) {
                    getContentFullLMsPerPredicate().putAll((Map<? extends String, ? extends SimpleLM>) o5);
                }
                System.out.println("done!");
            } catch (ClassNotFoundException | IOException ex) {
            } finally {
                try {
                    fin2.close();
                    fin3.close();
                    fin4.close();
                    fin5.close();
                } catch (IOException ex) {
                }
                try {
                    ois2.close();
                    ois3.close();
                    ois4.close();
                    ois5.close();
                } catch (IOException ex) {
                }
            }
        } else {
            return false;
        }
        return true;
    }

    /**
     *
     */
    public void writeLMs() {
        String file2 = "cache/wordLMs_SF_" + getDataset();
        String file3 = "cache/attrLMs_SF_" + getDataset();
        String file4 = "cache/wordFullLMs_SF_" + getDataset();
        String file5 = "cache/attrFullLMs_SF_" + getDataset();
        FileOutputStream fout2 = null;
        ObjectOutputStream oos2 = null;
        FileOutputStream fout3 = null;
        ObjectOutputStream oos3 = null;
        FileOutputStream fout4 = null;
        ObjectOutputStream oos4 = null;
        FileOutputStream fout5 = null;
        ObjectOutputStream oos5 = null;
        try {
            System.out.print("Write LMs...");
            fout2 = new FileOutputStream(file2);
            oos2 = new ObjectOutputStream(fout2);
            oos2.writeObject(getWordLMsPerPredicate());

            fout3 = new FileOutputStream(file3);
            oos3 = new ObjectOutputStream(fout3);
            oos3.writeObject(getContentLMsPerPredicate());

            fout4 = new FileOutputStream(file4);
            oos4 = new ObjectOutputStream(fout4);
            oos4.writeObject(getContentLMsPerPredicate());

            fout5 = new FileOutputStream(file5);
            oos5 = new ObjectOutputStream(fout5);
            oos5.writeObject(getContentLMsPerPredicate());
            System.out.println("done!");
        } catch (IOException ex) {
        } finally {
            try {
                fout2.close();
                fout3.close();
                fout4.close();
                fout5.close();
            } catch (IOException ex) {
            }
            try {
                oos2.close();
                oos3.close();
                oos4.close();
                oos5.close();
            } catch (IOException ex) {
            }
        }
    }

    @Override
    public boolean loadObservedAttrValues() {
        if (!isCache()) {
            return false;
        }
        String file = "cache/observedAttrValues_SF_" + getDataset();
        FileInputStream fin = null;
        ObjectInputStream ois = null;
        if ((new File(file)).exists()) {
            try {
                System.out.print("Load observed attrValue sequences...");

                fin = new FileInputStream(file);
                ois = new ObjectInputStream(fin);
                Object o = ois.readObject();
                if (getObservedAttrValueSequences() == null) {
                    if (o instanceof ArrayList) {
                        setObservedAttrValueSequences((ArrayList<ArrayList<String>>) o);
                    }
                } else if (o instanceof ArrayList) {
                    getObservedAttrValueSequences().addAll((ArrayList<ArrayList<String>>) o);
                }
                System.out.println("done!");
            } catch (ClassNotFoundException | IOException ex) {
            } finally {
                try {
                    fin.close();
                } catch (IOException ex) {
                }
                try {
                    ois.close();
                } catch (IOException ex) {
                }
            }
        } else {
            return false;
        }
        return true;
    }

    /**
     *
     */
    @Override
    public void writeObservedAttrValues() {
        String file = "cache/observedAttrValues_SF_" + getDataset();
        FileOutputStream fout = null;
        ObjectOutputStream oos = null;
        try {
            System.out.print("Write observed attrValue sequences...");
            fout = new FileOutputStream(file);
            oos = new ObjectOutputStream(fout);
            oos.writeObject(getObservedAttrValueSequences());
            System.out.println("done!");
        } catch (IOException ex) {
        } finally {
            try {
                fout.close();
            } catch (IOException ex) {
            }
            try {
                oos.close();
            } catch (IOException ex) {
            }
        }
    }

    /**
     *
     * @param dataSize
     * @param epoch
     * @param trainedAttrClassifiers
     * @param trainedWordClassifiers
     * @return
     */
    @Override
    public boolean loadClassifiers(int dataSize, int epoch, HashMap<String, JAROW> trainedAttrClassifiers, HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers) {
        if (!isCache()) {
            return false;
        }
        String file1 = "cache/attr_epoch=" + epoch + "_classifiers_" + getDataset() + "_" + dataSize;
        String file2 = "cache/word_epoch=" + epoch + "_classifiers_" + getDataset() + "_" + dataSize;
        FileInputStream fin1 = null;
        ObjectInputStream ois1 = null;
        FileInputStream fin2 = null;
        ObjectInputStream ois2 = null;
        if ((new File(file1)).exists()
                && (new File(file2)).exists()) {
            try {
                System.out.print("Load epoch=" + epoch + " classifiers...");
                fin1 = new FileInputStream(file1);
                ois1 = new ObjectInputStream(fin1);
                Object o1 = ois1.readObject();
                if (o1 instanceof HashMap) {
                    trainedAttrClassifiers.putAll((Map<? extends String, ? extends JAROW>) o1);
                }

                fin2 = new FileInputStream(file2);
                ois2 = new ObjectInputStream(fin2);
                Object o2 = ois2.readObject();
                if (o2 instanceof HashMap) {
                    trainedWordClassifiers.putAll((Map<? extends String, ? extends HashMap<String, JAROW>>) o2);
                }

                System.out.println("done!");
            } catch (ClassNotFoundException | IOException ex) {
            } finally {
                try {
                    fin1.close();
                    fin2.close();
                } catch (IOException ex) {
                }
                try {
                    ois1.close();
                    ois2.close();
                } catch (IOException ex) {
                }
            }
        } else {
            return false;
        }
        return true;
    }

    /**
     *
     * @param dataSize
     * @param epoch
     * @param trainedAttrClassifiers
     * @param trainedWordClassifiers
     */
    @Override
    public void writeClassifiers(int dataSize, int epoch, HashMap<String, JAROW> trainedAttrClassifiers, HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers) {
        String file1 = "cache/attr_epoch=" + epoch + "_classifiers_" + getDataset() + "_" + dataSize;
        String file2 = "cache/word_epoch=" + epoch + "_classifiers_" + getDataset() + "_" + dataSize;
        FileOutputStream fout1 = null;
        ObjectOutputStream oos1 = null;
        FileOutputStream fout2 = null;
        ObjectOutputStream oos2 = null;
        try {
            System.out.print("Write initial classifiers...");
            fout1 = new FileOutputStream(file1);
            oos1 = new ObjectOutputStream(fout1);
            oos1.writeObject(trainedAttrClassifiers);

            fout2 = new FileOutputStream(file2);
            oos2 = new ObjectOutputStream(fout2);
            oos2.writeObject(trainedWordClassifiers);

            System.out.println("done!");
        } catch (IOException ex) {
        } finally {
            try {
                fout1.close();
                fout2.close();
            } catch (IOException ex) {
            }
            try {
                oos1.close();
                oos2.close();
            } catch (IOException ex) {
            }
        }
    }

    @Override
    public Double evaluateGeneration(HashMap<String, JAROW> classifierAttrs,
            HashMap<String, HashMap<String, JAROW>> classifierWords, ArrayList<DatasetInstance> testingData,
            int epoch) {
        System.out.println("Evaluate argument generation ");

        ArrayList<ScoredFeaturizedTranslation<IString, String>> generations = new ArrayList<>();
        ConcurrentHashMap<DatasetInstance, ArrayList<Action>> generationActions = new ConcurrentHashMap<>();
        ArrayList<ArrayList<Sequence<IString>>> finalReferences = new ArrayList<>();
        ConcurrentHashMap<DatasetInstance, ArrayList<String>> finalReferencesWordSequences = new ConcurrentHashMap<>();
        ConcurrentHashMap<DatasetInstance, String> predictedWordSequences_overAllPredicates = new ConcurrentHashMap<>();
        ArrayList<String> allPredictedWordSequences = new ArrayList<>();
        ArrayList<String> allPredictedMRStr = new ArrayList<>();
        ArrayList<ArrayList<String>> allPredictedReferences = new ArrayList<>();
        HashMap<String, Double> attrCoverage = new HashMap<>();

        HashMap<String, HashSet<String>> abstractMRsToMRs = new HashMap<>();
        
        ExecutorService executor = Executors.newFixedThreadPool(THREAD_COUNT);
        
        ConcurrentHashMap<DatasetInstance, ArrayList<ScoredFeaturizedTranslation<IString, String>>> generationsConc = new ConcurrentHashMap<>();;
        ConcurrentHashMap<DatasetInstance, ArrayList<ArrayList<Sequence<IString>>>> finalReferencesConc = new ConcurrentHashMap<>();;
        ConcurrentHashMap<DatasetInstance, ArrayList<String>> allPredictedWordSequencesConc = new ConcurrentHashMap<>();;
        ConcurrentHashMap<DatasetInstance, ArrayList<String>> allPredictedMRStrConc = new ConcurrentHashMap<>();;
        ConcurrentHashMap<DatasetInstance, ArrayList<ArrayList<String>>> allPredictedReferencesConc = new ConcurrentHashMap<>();;
        ConcurrentHashMap<DatasetInstance, HashMap<String, Double>> attrCoverageConc = new ConcurrentHashMap<>();;
        ConcurrentHashMap<DatasetInstance, HashMap<String, HashSet<String>>> abstractMRsToMRsConc = new ConcurrentHashMap<>();;

        for (DatasetInstance di : testingData) {
            executor.execute(new EvaluatorThread(di, this, classifierAttrs, classifierWords, generationsConc, generationActions, finalReferencesConc, finalReferencesWordSequences, predictedWordSequences_overAllPredicates, allPredictedWordSequencesConc, allPredictedMRStrConc, allPredictedReferencesConc, attrCoverageConc, abstractMRsToMRsConc));
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
        for (DatasetInstance di : testingData) {
            generations.addAll(generationsConc.get(di));
            finalReferences.addAll(finalReferencesConc.get(di));
            allPredictedWordSequences.addAll(allPredictedWordSequencesConc.get(di));
            allPredictedMRStr.addAll(allPredictedMRStrConc.get(di));
            allPredictedReferences.addAll(allPredictedReferencesConc.get(di));
            attrCoverage.putAll(attrCoverageConc.get(di));
            abstractMRsToMRs.putAll(abstractMRsToMRsConc.get(di));
        }

        @SuppressWarnings({"unchecked", "rawtypes"})
        BLEUMetric BLEU = new BLEUMetric(finalReferences, 4, false);
        @SuppressWarnings("unchecked")
        Double bleuScore = BLEU.score(generations);

        double finalCoverageError = 0.0;
        finalCoverageError = attrCoverage.values().stream().map((c) -> c).reduce(finalCoverageError, (accumulator, _item) -> accumulator + _item);
        finalCoverageError /= attrCoverage.size();
        for (int i = 0; i < allPredictedWordSequences.size(); i++) {
            double maxRouge = 0.0;
            String predictedWordSequence = allPredictedWordSequences.get(i).replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
            for (String ref : allPredictedReferences.get(i)) {
                double rouge = Rouge.ROUGE_N(predictedWordSequence, ref, 4);
                if (rouge > maxRouge) {
                    maxRouge = rouge;
                }
            }
            //System.out.println(allPredictedMRStr.get(i) + "\t" + maxRouge + "\t" + allPredictedWordSequences.get(i) + "\t" + refs);
        }

        double avgRougeScore = 0.0;
        String detailedRes = "";

        avgRougeScore = testingData.stream().map((di) -> {
            double maxRouge = 0.0;
            if (!finalReferencesWordSequences.containsKey(di)) {
                System.out.println(di.getMeaningRepresentation().getAbstractMR());
            }
            String predictedWordSequence = predictedWordSequences_overAllPredicates.get(di).replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
            for (String ref : finalReferencesWordSequences.get(di)) {
                double rouge = Rouge.ROUGE_N(predictedWordSequence, ref, 4);
                if (rouge > maxRouge) {
                    maxRouge = rouge;
                }
            }
            return maxRouge;
        }).map((maxRouge) -> maxRouge).reduce(avgRougeScore, (accumulator, _item) -> accumulator + _item);
        System.out.println("BLEU: \t" + bleuScore);
        //System.out.println("g: " + generations);
        //System.out.println("attr: " + predictedAttrLists);
        //System.out.println("BLEU smooth: \t" + bleuSmoothScore);
        //System.out.println("g: " + generations);
        //System.out.println("attr: " + predictedAttrLists);
        //System.out.println("BLEU smooth: \t" + bleuSmoothScore);
        System.out.println("ROUGE: \t" + (avgRougeScore / allPredictedWordSequences.size()));
        System.out.println("COVERAGE ERROR: \t" + finalCoverageError);
        System.out.println("BRC: \t" + ((avgRougeScore / allPredictedWordSequences.size()) + bleuScore + (1.0 - finalCoverageError)) / 3.0);

        if (isCalculateResultsPerPredicate()) {
            ////////////////////////
            //ArrayList<String> bestPredictedStrings = new ArrayList<>();
            //ArrayList<String> bestPredictedStringsMRs = new ArrayList<>();
            double uniqueMRsInTestAndNotInTrainAllPredWordBLEU = 0.0;
            double uniqueMRsInTestAndNotInTrainAllPredWordROUGE = 0.0;
            double uniqueMRsInTestAndNotInTrainAllPredWordCOVERAGEERR = 0.0;
            double uniqueMRsInTestAndNotInTrainAllPredWordBRC = 0.0;

            detailedRes = "";
            ArrayList<DatasetInstance> abstractMRList = new ArrayList<>();
            HashSet<String> reportedAbstractMRs = new HashSet<>();
            testingData.stream().filter((di) -> (!reportedAbstractMRs.contains(di.getMeaningRepresentation().getAbstractMR()))).map((di) -> {
                reportedAbstractMRs.add(di.getMeaningRepresentation().getAbstractMR());
                return di;
            }).forEachOrdered((di) -> {
                boolean isInTraining = false;
                for (DatasetInstance di2 : getTrainingData()) {
                    if (di2.getMeaningRepresentation().getAbstractMR().equals(di.getMeaningRepresentation().getAbstractMR())) {
                        isInTraining = true;
                    }
                }
                if (!isInTraining) {
                    for (DatasetInstance di2 : getValidationData()) {
                        if (di2.getMeaningRepresentation().getAbstractMR().equals(di.getMeaningRepresentation().getAbstractMR())) {
                            isInTraining = true;
                        }
                    }
                }
                if (!isInTraining) {
                    abstractMRList.add(di);
                }
            });
            for (DatasetInstance di : abstractMRList) {
                Double bestROUGE = -100.0;
                Double bestBLEU = -100.0;
                Double bestCover = -100.0;
                Double bestHarmonicMean = -100.0;
                String predictedString = predictedWordSequences_overAllPredicates.get(di);
                reportedAbstractMRs.add(di.getMeaningRepresentation().getAbstractMR());
                double maxRouge = 0.0;
                String predictedWordSequence = predictedString.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
                for (String ref : finalReferencesWordSequences.get(di)) {
                    double rouge = Rouge.ROUGE_N(predictedWordSequence, ref, 4);
                    if (rouge > maxRouge) {
                        maxRouge = rouge;
                    }
                }

                double BLEUSmooth = BLEUMetric.computeLocalSmoothScore(predictedWordSequence, finalReferencesWordSequences.get(di), 4);
                double cover = 1.0 - attrCoverage.get(predictedString);
                double harmonicMean = 3.0 / (1.0 / BLEUSmooth + 1.0 / maxRouge + 1.0 / cover);

                if (harmonicMean > bestHarmonicMean) {
                    bestROUGE = maxRouge;
                    bestBLEU = BLEUSmooth;
                    bestCover = cover;
                    bestHarmonicMean = harmonicMean;
                }

                uniqueMRsInTestAndNotInTrainAllPredWordBLEU += bestBLEU;
                uniqueMRsInTestAndNotInTrainAllPredWordROUGE += bestROUGE;
                uniqueMRsInTestAndNotInTrainAllPredWordCOVERAGEERR += bestCover;
                uniqueMRsInTestAndNotInTrainAllPredWordBRC += bestHarmonicMean;
            }
            uniqueMRsInTestAndNotInTrainAllPredWordBLEU /= abstractMRList.size();
            uniqueMRsInTestAndNotInTrainAllPredWordROUGE /= abstractMRList.size();
            uniqueMRsInTestAndNotInTrainAllPredWordCOVERAGEERR /= abstractMRList.size();
            uniqueMRsInTestAndNotInTrainAllPredWordBRC /= abstractMRList.size();
            System.out.println("UNIQUE (NOT IN TRAIN) WORD ALL PRED BLEU: \t" + uniqueMRsInTestAndNotInTrainAllPredWordBLEU);
            System.out.println("UNIQUE (NOT IN TRAIN) WORD ALL PRED ROUGE: \t" + uniqueMRsInTestAndNotInTrainAllPredWordROUGE);
            System.out.println("UNIQUE (NOT IN TRAIN) WORD ALL PRED COVERAGE ERROR: \t" + (1.0 - uniqueMRsInTestAndNotInTrainAllPredWordCOVERAGEERR));
            System.out.println("UNIQUE (NOT IN TRAIN) WORD ALL PRED BRC: \t" + uniqueMRsInTestAndNotInTrainAllPredWordBRC);

            abstractMRList.forEach((di) -> {
                System.out.println(di.getMeaningRepresentation().getAbstractMR() + "\t" + predictedWordSequences_overAllPredicates.get(di));
            });
            System.out.println("TOTAL SET SIZE: \t" + abstractMRList.size());
            //System.out.println(abstractMRList);  
            //System.out.println(detailedRes);
        }
        ArrayList<String> bestPredictedStrings = new ArrayList<>();
        ArrayList<String> bestPredictedStringsMRs = new ArrayList<>();
        double uniqueAllPredWordBLEU = 0.0;
        double uniqueAllPredWordROUGE = 0.0;
        double uniqueAllPredWordCOVERAGEERR = 0.0;
        double uniqueAllPredWordBRC = 0.0;

        HashSet<String> reportedAbstractMRs = new HashSet<>();
        for (DatasetInstance di : testingData) {
            if (!reportedAbstractMRs.contains(di.getMeaningRepresentation().getAbstractMR())) {
                String bestPredictedString = "";
                Double bestROUGE = -100.0;
                Double bestBLEU = -100.0;
                Double bestCover = -100.0;
                Double bestHarmonicMean = -100.0;
                String predictedString = predictedWordSequences_overAllPredicates.get(di);
                reportedAbstractMRs.add(di.getMeaningRepresentation().getAbstractMR());
                double maxRouge = 0.0;
                String predictedWordSequence = predictedString.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
                for (String ref : finalReferencesWordSequences.get(di)) {
                    double rouge = Rouge.ROUGE_N(predictedWordSequence, ref, 4);
                    if (rouge > maxRouge) {
                        maxRouge = rouge;
                    }
                }

                double BLEUSmooth = BLEUMetric.computeLocalSmoothScore(predictedWordSequence, finalReferencesWordSequences.get(di), 4);
                double cover = 1.0 - attrCoverage.get(predictedString);
                double harmonicMean = 3.0 / (1.0 / BLEUSmooth + 1.0 / maxRouge + 1.0 / cover);

                if (harmonicMean > bestHarmonicMean) {
                    bestPredictedString = predictedString;
                    bestROUGE = maxRouge;
                    bestBLEU = BLEUSmooth;
                    bestCover = cover;
                    bestHarmonicMean = harmonicMean;
                }
                bestPredictedStrings.add(bestPredictedString);
                bestPredictedStringsMRs.add(di.getMeaningRepresentation().getMRstr());

                uniqueAllPredWordBLEU += bestBLEU;
                uniqueAllPredWordROUGE += bestROUGE;
                uniqueAllPredWordCOVERAGEERR += bestCover;
                uniqueAllPredWordBRC += bestHarmonicMean;
            }
            //}
        }
        if (isCalculateResultsPerPredicate()) {
            uniqueAllPredWordBLEU /= reportedAbstractMRs.size();
            uniqueAllPredWordROUGE /= reportedAbstractMRs.size();
            uniqueAllPredWordCOVERAGEERR /= reportedAbstractMRs.size();
            uniqueAllPredWordBRC /= reportedAbstractMRs.size();
            System.out.println("UNIQUE WORD ALL PRED BLEU: \t" + uniqueAllPredWordBLEU);
            System.out.println("UNIQUE WORD ALL PRED ROUGE: \t" + uniqueAllPredWordROUGE);
            System.out.println("UNIQUE WORD ALL PRED COVERAGE ERROR: \t" + (1.0 - uniqueAllPredWordCOVERAGEERR));
            System.out.println("UNIQUE WORD ALL PRED BRC: \t" + uniqueAllPredWordBRC);
            System.out.println(detailedRes);
            System.out.println("TOTAL: \t" + reportedAbstractMRs.size());

            ////////////////////////
            for (String predicate : getPredicates()) {
                detailedRes = "";
                bestPredictedStrings = new ArrayList<>();
                bestPredictedStringsMRs = new ArrayList<>();
                double uniquePredWordBLEU = 0.0;
                double uniquePredWordROUGE = 0.0;
                double uniquePredWordCOVERAGEERR = 0.0;
                double uniquePredWordBRC = 0.0;

                reportedAbstractMRs = new HashSet<>();
                for (DatasetInstance di : testingData) {
                    if (di.getMeaningRepresentation().getPredicate().equals(predicate)
                            && !reportedAbstractMRs.contains(di.getMeaningRepresentation().getAbstractMR())) {
                        String bestPredictedString = "";
                        Double bestROUGE = -100.0;
                        Double bestBLEU = -100.0;
                        Double bestCover = -100.0;
                        Double bestHarmonicMean = -100.0;

                        String predictedString = predictedWordSequences_overAllPredicates.get(di);
                        reportedAbstractMRs.add(di.getMeaningRepresentation().getAbstractMR());
                        double maxRouge = 0.0;
                        String predictedWordSequence = predictedString.replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
                        for (String ref : finalReferencesWordSequences.get(di)) {
                            double rouge = Rouge.ROUGE_N(predictedWordSequence, ref, 4);
                            if (rouge > maxRouge) {
                                maxRouge = rouge;
                            }
                        }

                        double BLEUSmooth = BLEUMetric.computeLocalSmoothScore(predictedWordSequence, finalReferencesWordSequences.get(di), 4);
                        double cover = 1.0 - attrCoverage.get(predictedString);
                        double harmonicMean = 3.0 / (1.0 / BLEUSmooth + 1.0 / maxRouge + 1.0 / cover);

                        if (harmonicMean > bestHarmonicMean) {
                            bestPredictedString = predictedString;
                            bestROUGE = maxRouge;
                            bestBLEU = BLEUSmooth;
                            bestCover = cover;
                            bestHarmonicMean = harmonicMean;
                        }
                        bestPredictedStrings.add(bestPredictedString);
                        bestPredictedStringsMRs.add(di.getMeaningRepresentation().getMRstr());

                        uniquePredWordBLEU += bestBLEU;
                        uniquePredWordROUGE += bestROUGE;
                        uniquePredWordCOVERAGEERR += bestCover;
                        uniquePredWordBRC += bestHarmonicMean;
                    }
                }

                uniquePredWordBLEU /= reportedAbstractMRs.size();
                uniquePredWordROUGE /= reportedAbstractMRs.size();
                uniquePredWordCOVERAGEERR /= reportedAbstractMRs.size();
                uniquePredWordBRC /= reportedAbstractMRs.size();
                System.out.println("UNIQUE WORD " + predicate + " BLEU: \t" + uniquePredWordBLEU);
                System.out.println("UNIQUE WORD " + predicate + " ROUGE: \t" + uniquePredWordROUGE);
                System.out.println("UNIQUE WORD " + predicate + " COVERAGE ERROR: \t" + (1.0 - uniquePredWordCOVERAGEERR));
                System.out.println("UNIQUE WORD " + predicate + " BRC: \t" + uniquePredWordBRC);
                System.out.println(detailedRes);
                System.out.println("TOTAL " + predicate + ": \t" + reportedAbstractMRs.size());
            }
        }

        BufferedWriter bw = null;
        File f = null;

        BufferedWriter bw2 = null;
        File f2 = null;
        try {
            f = new File("results/E2E" + "TextsAfter" + (epoch) + "_" + JLOLS.sentenceCorrectionFurtherSteps + "_" + JLOLS.p + "epochsTESTINGDATA.txt");
            f2 = new File("results/E2E" + "distinctTextsAfter" + (epoch) + "_" + JLOLS.sentenceCorrectionFurtherSteps + "_" + JLOLS.p + "epochsTESTINGDATA.txt");
        } catch (NullPointerException e) {
        }

        try {
            bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f)));
            bw2 = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f2)));
        } catch (FileNotFoundException e) {
        }

        try {
            bw.write("BLEU:" + bleuScore);
            bw.write("\n");
        } catch (IOException e) {
        }

        //Unique according to the MR as it appears on the dataset (for evaluation consistency)
        ArrayList<DatasetInstance> uniqueDi = new ArrayList<>();
        //Distinct according to the abstract MR
        ArrayList<DatasetInstance> distinctDi = new ArrayList<>();
        HashMap<String, String> exportMap = new HashMap<>();        
        for (DatasetInstance di : testingData) {       
            boolean distinct = true;
            for (DatasetInstance tr : getTrainingData()) {
                if (tr.getMeaningRepresentation().getAbstractMR().equals(di.getMeaningRepresentation().getAbstractMR())) {
                    distinct = false;
                }
            }
            String ex = predictedWordSequences_overAllPredicates.get(di).replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
            if (!exportMap.containsKey(di.getMeaningRepresentation().getMRstr())) {
                uniqueDi.add(di);
                if (distinct) {
                    distinctDi.add(di);
                }
                exportMap.put(di.getMeaningRepresentation().getMRstr(), ex);
            } else {
                double BLEUSmoothCurrent = BLEUMetric.computeLocalSmoothScore(exportMap.get(di.getMeaningRepresentation().getMRstr()), finalReferencesWordSequences.get(di), 4);
                double BLEUSmoothNew = BLEUMetric.computeLocalSmoothScore(ex, finalReferencesWordSequences.get(di), 4);
                if (BLEUSmoothNew > BLEUSmoothCurrent) {
                    exportMap.put(di.getMeaningRepresentation().getMRstr(), ex);
                }
            }
        } 
        for (DatasetInstance di : uniqueDi) {
            try {
                bw.write(exportMap.get(di.getMeaningRepresentation().getMRstr()));
                bw.write("\n");
            } catch (IOException e) {
            }
        }
        for (DatasetInstance di : distinctDi) {
            try {
                bw2.write(exportMap.get(di.getMeaningRepresentation().getMRstr()));
                bw2.write("\n");
            } catch (IOException e) {
            }
        }

        try {
            bw.close();
            bw2.close();
        } catch (IOException e) {
        }
        return bleuScore;
    }

    @Override
    public void createNaiveAlignments(ArrayList<DatasetInstance> trainingData) {
        HashMap<String, HashMap<ArrayList<Action>, HashMap<Action, Integer>>> punctPatterns = new HashMap<>();
        getPredicates().forEach((predicate) -> {
            punctPatterns.put(predicate, new HashMap<ArrayList<Action>, HashMap<Action, Integer>>());//punctPattern key:predicate
        });
        HashMap<DatasetInstance, ArrayList<Action>> punctRealizations = new HashMap<DatasetInstance, ArrayList<Action>>();

        trainingData.stream().map((di) -> {
            HashMap<ArrayList<Action>, ArrayList<Action>> calculatedRealizationsCache = new HashMap<>();//key directReferenceSequence
            HashSet<ArrayList<Action>> initRealizations = new HashSet<>();
            if (!calculatedRealizationsCache.containsKey(di.getDirectReferenceSequence())) {
                initRealizations.add(di.getDirectReferenceSequence());
            }
            initRealizations.stream().map((realization) -> {
                HashMap<String, HashSet<String>> values = new HashMap<>();
                di.getMeaningRepresentation().getAttributeValues().keySet().forEach((attr) -> {
                    values.put(attr, new HashSet<>(di.getMeaningRepresentation().getAttributeValues().get(attr)));//attr to value
                });
                ArrayList<Action> randomRealization = new ArrayList<>();
                for (int i = 0; i < realization.size(); i++) {
                    Action a = realization.get(i);
                    if (a.getAttribute().equals(Action.TOKEN_PUNCT) || a.getWord().matches("[:,.?!;']")) {

                        randomRealization.add(new Action(a.getWord(), a.getAttribute()));//only record the punct
                    } else {
                        randomRealization.add(new Action(a.getWord(), ""));
                    }

                }

                if (values.keySet().isEmpty()) {
                    for (int i = 0; i < randomRealization.size(); i++) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!getAttributes().get(di.getMeaningRepresentation().getPredicate()).contains("empty")) {
                                getAttributes().get(di.getMeaningRepresentation().getPredicate()).add("empty");
                            }
                            randomRealization.get(i).setAttribute("empty=empty");
                        }
                    }
                } else {
                    HashMap<Double, HashMap<String, ArrayList<Integer>>> indexAlignments = new HashMap<>();
                    //HashSet<String> noValueAttrs = new HashSet<String>();
                    values.keySet().forEach((attr) -> {
                        values.get(attr).stream().filter((value) -> ((!value.startsWith(Action.TOKEN_X))
                                && !value.isEmpty())).map((value) -> {
                            String valueToCheck = value;
                            //if(valueToCheck.contains("familyfriendly")){
                            //valueToCheck = valueToCheck.replace("familyfriendly", "family friendly");
                            //}
                            /*
                            if (valueToCheck.equals("no")
                                    || valueToCheck.equals("yes")
                                    || valueToCheck.equals("yes or no")
                                    || valueToCheck.equals("none")
                                    //|| attr.equals("dont_care")
                                    || valueToCheck.equals("empty")) {
                                valueToCheck = attr + ":" + value;
                                noValueAttrs.add(attr + "=" + value);
                            }
                            if (valueToCheck.equals(attr)) {
                                noValueAttrs.add(attr + "=" + value);
                            }*/
                            return valueToCheck;
                        }).filter((valueToCheck) -> (!valueToCheck.equals("empty:empty")
                                && getValueAlignments().containsKey(valueToCheck))).forEachOrdered((valueToCheck) -> {
                            for (ArrayList<String> align : getValueAlignments().get(valueToCheck).keySet()) {
                                int n = align.size();
                                for (int i = 0; i <= randomRealization.size() - n; i++) {
                                    ArrayList<String> compare = new ArrayList<String>();
                                    ArrayList<Integer> indexAlignment = new ArrayList<Integer>();
                                    for (int j = 0; j < n; j++) {
                                        compare.add(randomRealization.get(i + j).getWord());
                                        indexAlignment.add(i + j);
                                    }
                                    if (compare.equals(align)) {
                                        if (!indexAlignments.containsKey(getValueAlignments().get(valueToCheck).get(align))) {
                                            indexAlignments.put(getValueAlignments().get(valueToCheck).get(align), new HashMap<>());
                                        }
                                        indexAlignments.get(getValueAlignments().get(valueToCheck).get(align)).put(attr + "=" + valueToCheck, indexAlignment);
                                    }
                                }
                            }
                        });
                    });
                    ArrayList<Double> similarities = new ArrayList<>(indexAlignments.keySet());
                    Collections.sort(similarities);
                    HashSet<String> assignedAttrValues = new HashSet<String>();
                    HashSet<Integer> assignedIntegers = new HashSet<Integer>();
                    for (int i = similarities.size() - 1; i >= 0; i--) {
                        for (String attrValue : indexAlignments.get(similarities.get(i)).keySet()) {
                            if (!assignedAttrValues.contains(attrValue)) {
                                boolean isUnassigned = true;
                                for (Integer index : indexAlignments.get(similarities.get(i)).get(attrValue)) {
                                    if (assignedIntegers.contains(index)) {
                                        isUnassigned = false;
                                    }
                                }
                                if (isUnassigned) {
                                    assignedAttrValues.add(attrValue);
                                    for (Integer index : indexAlignments.get(similarities.get(i)).get(attrValue)) {
                                        assignedIntegers.add(index);
                                        randomRealization.get(index).setAttribute(attrValue.toLowerCase().trim());
                                    }
                                }
                            }
                        }
                    }//value alignment part has been assigned

                    //System.out.println("-1: " + randomRealization);
                    randomRealization.stream().filter((a) -> (a.getWord().startsWith(Action.TOKEN_X))).forEachOrdered((a) -> {
                        String attr = a.getWord().substring(3, a.getWord().lastIndexOf('_')).toLowerCase().trim();
                        a.setAttribute(attr + "=" + a.getWord());
                    });/*
                    HashSet<String> unalignedNoValueAttrs = new HashSet<>();
                    noValueAttrs.forEach((noValueAttr) -> {
                        boolean assigned = false;
                        for (Action a : randomRealization) {
                            if (a.getAttribute().equals(noValueAttr)) {
                                assigned = true;
                            }
                        }
                        if (!assigned) {
                            unalignedNoValueAttrs.add(noValueAttr);
                        }
                    });
                    boolean isAllEmpty = true;
                    boolean hasSpace = false;
                    for (int i = 0; i < randomRealization.size(); i++) {
                        if (!randomRealization.get(i).getAttribute().isEmpty()
                                && !randomRealization.get(i).getAttribute().equals("[]")
                                && !randomRealization.get(i).getAttribute().equals(Action.TOKEN_PUNCT)) {
                            isAllEmpty = false;
                        }
                        if (randomRealization.get(i).getAttribute().isEmpty()
                                || randomRealization.get(i).getAttribute().equals("[]")) {
                            hasSpace = true;
                        }
                    }
                    if (isAllEmpty && hasSpace && !unalignedNoValueAttrs.isEmpty()) {
                        unalignedNoValueAttrs.forEach((attrValue) -> {
                            int index = getRandomGen().nextInt(randomRealization.size());
                            boolean change = false;
                            while (!change) {
                                if (!randomRealization.get(index).getAttribute().equals(Action.TOKEN_PUNCT)) {
                                    randomRealization.get(index).setAttribute(attrValue.toLowerCase().trim());
                                    change = true;
                                } else {
                                    index = getRandomGen().nextInt(randomRealization.size());
                                }
                            }
                        });
                    }*/
                    //System.out.println(isAllEmpty + " " + hasSpace + " " + unalignedNoValueAttrs);
                    //System.out.println(">> " + noValueAttrs);
                    //System.out.println(">> " + values);
                    //System.out.println("0: " + randomRealization);
                    String previousAttr = "";
                    int start = -1;
                    for (int i = 0; i < randomRealization.size(); i++) {
                        if (!randomRealization.get(i).getAttribute().equals(Action.TOKEN_PUNCT)
                                && !randomRealization.get(i).getAttribute().isEmpty()
                                && !randomRealization.get(i).getAttribute().equals("[]")) {
                            if (start != -1) {
                                int middle = (start + i - 1) / 2 + 1;
                                for (int j = start; j < middle; j++) {
                                    if (randomRealization.get(j).getAttribute().isEmpty()
                                            || randomRealization.get(j).getAttribute().equals("[]")) {
                                        randomRealization.get(j).setAttribute(previousAttr);
                                    }
                                }
                                for (int j = middle; j < i; j++) {
                                    if (randomRealization.get(j).getAttribute().isEmpty()
                                            || randomRealization.get(j).getAttribute().equals("[]")) {
                                        randomRealization.get(j).setAttribute(randomRealization.get(i).getAttribute());
                                    }
                                }
                            }
                            start = i;
                            previousAttr = randomRealization.get(i).getAttribute();
                        } else {
                            previousAttr = "";
                        }
                    }
                    //System.out.println("1: " + randomRealization);
                    previousAttr = "";
                    for (int i = randomRealization.size() - 1; i >= 0; i--) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!previousAttr.isEmpty()) {
                                randomRealization.get(i).setAttribute(previousAttr);
                            }
                        } else if (!randomRealization.get(i).getAttribute().equals(Action.TOKEN_PUNCT)) {
                            previousAttr = randomRealization.get(i).getAttribute();
                        } else {
                            previousAttr = "";
                        }
                    }
                    //System.out.println("2: " + randomRealization);
                    previousAttr = "";
                    for (int i = 0; i < randomRealization.size(); i++) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!previousAttr.isEmpty()) {
                                randomRealization.get(i).setAttribute(previousAttr);
                            }
                        } else if (!randomRealization.get(i).getAttribute().equals(Action.TOKEN_PUNCT)) {
                            previousAttr = randomRealization.get(i).getAttribute();
                        }
                    }
                    //System.out.println("3: " + randomRealization);
                    previousAttr = "";
                    for (int i = randomRealization.size() - 1; i >= 0; i--) {
                        if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                            if (!previousAttr.isEmpty()) {
                                randomRealization.get(i).setAttribute(previousAttr);
                            }
                        } else if (!randomRealization.get(i).getAttribute().equals(Action.TOKEN_PUNCT)) {
                            previousAttr = randomRealization.get(i).getAttribute();
                        }
                    }
                    //System.out.println("4: " + randomRealization);
                }
                //FIX WRONG @PUNCT@

                String previousAttr = "";/*
                for (int i = randomRealization.size() - 1; i >= 0; i--) {
                    if (randomRealization.get(i).getAttribute().equals(Action.TOKEN_PUNCT) && !randomRealization.get(i).getWord().matches("[,.?!;:']")) {
                        if (!previousAttr.isEmpty()) {
                            randomRealization.get(i).setAttribute(previousAttr);
                        }
                    } else if (!randomRealization.get(i).getAttribute().equals(Action.TOKEN_PUNCT)) {
                        previousAttr = randomRealization.get(i).getAttribute();
                    }
                }*/
                ArrayList<Action> cleanRandomRealization = new ArrayList<>();
                randomRealization.stream().filter((a) -> (!a.getAttribute().equals(Action.TOKEN_PUNCT)) && !a.getWord().matches("[,.:;'?!]")).forEachOrdered((a) -> {
                    cleanRandomRealization.add(a);//no punctuation in the action (clean) 
                });

                //ADD END TOKENS
                ArrayList<Action> endRandomRealization = new ArrayList<>();
                previousAttr = "";
                for (int i = 0; i < cleanRandomRealization.size(); i++) {
                    Action a = cleanRandomRealization.get(i);
                    if (!previousAttr.isEmpty()
                            && !a.getAttribute().equals(previousAttr)) {
                        endRandomRealization.add(new Action(Action.TOKEN_END, previousAttr));
                    }
                    endRandomRealization.add(a);
                    previousAttr = a.getAttribute();
                }
                endRandomRealization.add(new Action(Action.TOKEN_END, previousAttr));
                endRandomRealization.add(new Action(Action.TOKEN_END, Action.TOKEN_END));
                calculatedRealizationsCache.put(realization, endRandomRealization);
                //System.out.println(di.getMeaningRepresentation().getPredicate() + ": " + endRandomRealization);
                ArrayList<String> attrValues = new ArrayList<String>();
                endRandomRealization.forEach((a) -> {
                    if (attrValues.isEmpty()) {
                        attrValues.add(a.getAttribute());
                    } else if (!attrValues.get(attrValues.size() - 1).equals(a.getAttribute())) {
                        attrValues.add(a.getAttribute());
                    }
                });
                if (attrValues.size() > getMaxContentSequenceLength()) {
                    setMaxContentSequenceLength(attrValues.size());
                }
                ArrayList<Action> punctRealization = new ArrayList<>();
                punctRealization.addAll(randomRealization);
                previousAttr = "";
                for (int i = 0; i < punctRealization.size(); i++) {
                    if (!punctRealization.get(i).getAttribute().equals(Action.TOKEN_PUNCT)) {
                        if (!punctRealization.get(i).getAttribute().equals(previousAttr)
                                && !previousAttr.isEmpty()) {
                            punctRealization.add(i, new Action(Action.TOKEN_END, previousAttr));
                            i++;
                        }
                        previousAttr = punctRealization.get(i).getAttribute();
                    }
                }
                if (!punctRealization.get(punctRealization.size() - 1).getWord().equals(Action.TOKEN_END)) {
                    punctRealization.add(new Action(Action.TOKEN_END, previousAttr));
                }
                for (Action a : punctRealization) {
                    if (a.getWord().matches("[,.:;'?!]")) {
                        a.setAttribute(Action.TOKEN_PUNCT);
                    }
                }
                return punctRealization;
            }).map((punctRealization) -> {
                punctRealizations.put(di, punctRealization);
                return punctRealization;
            }).forEachOrdered((punctRealization) -> {
                for (int i = 0; i < punctRealization.size(); i++) {
                    Action a = punctRealization.get(i);
                    if (a.getAttribute().equals(Action.TOKEN_PUNCT)) {// find punctuation surrounding actions
                        boolean legal = true;
                        ArrayList<Action> surroundingActions = new ArrayList<>();
                        if (i - 2 >= 0) {
                            surroundingActions.add(punctRealization.get(i - 2));
                        } else {
                            surroundingActions.add(null);
                        }
                        if (i - 1 >= 0) {
                            surroundingActions.add(punctRealization.get(i - 1));
                        } else {
                            legal = false;
                        }
                        boolean oneMore = false;
                        if (i + 1 < punctRealization.size()) {
                            surroundingActions.add(punctRealization.get(i + 1));
                            if (!punctRealization.get(i + 1).getAttribute().equals(Action.TOKEN_END)) {
                                oneMore = true;
                            }
                        } else {
                            legal = false;
                        }
                        if (oneMore && i + 2 < punctRealization.size()) {
                            surroundingActions.add(punctRealization.get(i + 2));
                        } else {
                            surroundingActions.add(null);
                        }
                        if (legal) {
                            if (!punctPatterns.get(di.getMeaningRepresentation().getPredicate()).containsKey(surroundingActions)) {
                                punctPatterns.get(di.getMeaningRepresentation().getPredicate()).put(surroundingActions, new HashMap<Action, Integer>());
                            }
                            if (!punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surroundingActions).containsKey(a)) {
                                punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surroundingActions).put(a, 1);
                            } else {
                                punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surroundingActions).put(a, punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surroundingActions).get(a) + 1);
                            }
                        }
                    }
                }
            });
            di.setDirectReferenceSequence(calculatedRealizationsCache.get(di.getDirectReferenceSequence()));
            return di;
        }).forEachOrdered((di) -> {

            HashSet<String> attrValuesToBeMentioned = new HashSet<>();
            di.getMeaningRepresentation().getAttributeValues().keySet().forEach((attribute) -> {
                int a = 0;
                for (String value : di.getMeaningRepresentation().getAttributeValues().get(attribute)) {
                    if (value.startsWith("\"x")) {
                        value = "x" + a;
                        a++;
                    } else if (value.startsWith("\"")) {
                        value = value.substring(1, value.length() - 1).replaceAll(" ", "_");
                    }
                    attrValuesToBeMentioned.add(attribute + "=" + value);
                }
            });
            di.getDirectReferenceSequence().stream().map((key) -> {
                attrValuesToBeMentioned.remove(key.getAttribute());//not aligned in the realization
                return key;
            });

        });
        punctRealizations.keySet().forEach((di) -> {
            ArrayList<Action> punctRealization = punctRealizations.get(di);
            punctPatterns.get(di.getMeaningRepresentation().getPredicate()).keySet().forEach((surrounds) -> {
                int beforeNulls = 0;
                if (surrounds.get(0) == null) {
                    beforeNulls++;
                }
                if (surrounds.get(1) == null) {
                    beforeNulls++;
                }
                for (int i = 0 - beforeNulls; i < punctRealization.size(); i++) {
                    boolean matches = true;
                    int m = 0;
                    for (int s = 0; s < surrounds.size(); s++) {
                        if (surrounds.get(s) != null) {
                            if (i + s < punctRealization.size()) {
                                if (!punctRealization.get(i + s).getWord().equals(surrounds.get(s).getWord()) /*|| !cleanActionList.get(i).getAttribute().equals(surrounds.get(s).getAttribute())*/) {
                                    matches = false;
                                    s = surrounds.size();
                                } else {
                                    m++;
                                }
                            } else {
                                matches = false;
                                s = surrounds.size();
                            }
                        } else if (s < 2 && i + s >= 0) {
                            matches = false;
                            s = surrounds.size();
                        } else if (s >= 2 && i + s < punctRealization.size()) {
                            matches = false;
                            s = surrounds.size();
                        }
                    }
                    if (matches && m > 0) {
                        Action a = new Action("", "");
                        if (!punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surrounds).containsKey(a)) {
                            punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surrounds).put(a, 1);
                        } else {
                            punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surrounds).put(a, punctPatterns.get(di.getMeaningRepresentation().getPredicate()).get(surrounds).get(a) + 1);
                        }
                    }
                }
            });
        });
        punctPatterns.keySet().forEach((predicate) -> {
            punctPatterns.get(predicate).keySet().forEach((punct) -> {
                Action bestAction = null;
                int bestCount = 0;
                for (Action a : punctPatterns.get(predicate).get(punct).keySet()) {
                    if (punctPatterns.get(predicate).get(punct).get(a) > bestCount) {
                        bestAction = a;
                        bestCount = punctPatterns.get(predicate).get(punct).get(a);
                    } else if (punctPatterns.get(predicate).get(punct).get(a) == bestCount
                            && bestAction.getWord().isEmpty()) {
                        bestAction = a;
                    }
                }
                if (!getPunctuationPatterns().containsKey(predicate)) {
                    getPunctuationPatterns().put(predicate, new HashMap<ArrayList<Action>, Action>());
                }
                if (!bestAction.getWord().isEmpty()) {
                    getPunctuationPatterns().get(predicate).put(punct, bestAction);
                }
            });
        });
    }

    @Override
    public Instance createContentInstance(String predicate, String bestAction, ArrayList<String> previousGeneratedAttrs,
            HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesToBeMentioned,
            MeaningRepresentation MR, HashMap<String, HashSet<String>> availableAttributeActions) {
        TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

        if (!bestAction.isEmpty()) {
            //COSTS
            if (bestAction.equals(Action.TOKEN_END)) {
                costs.put(Action.TOKEN_END, 0.0);
                availableAttributeActions.get(predicate).forEach((action) -> {
                    costs.put(action, 1.0);
                });
            } else if (!bestAction.contains("@TOK@")) {
                costs.put(Action.TOKEN_END, 1.0);
                availableAttributeActions.get(predicate).forEach((action) -> {
                    String attr = bestAction;
                    if (bestAction.contains("=")) {
                        attr = bestAction.substring(0, bestAction.indexOf('=')).toLowerCase().trim();
                    }
                    if (action.equals(attr)) {
                        costs.put(action, 0.0);
                    } else {
                        costs.put(action, 1.0);
                    }
                });
            }
        }

        return createContentInstanceWithCosts(predicate, costs, previousGeneratedAttrs, attrValuesAlreadyMentioned, attrValuesToBeMentioned, availableAttributeActions, MR);
    }

    @Override
    public Instance createContentInstanceWithCosts(String predicate, TObjectDoubleHashMap<String> costs,
            ArrayList<String> previousGeneratedAttrs, HashSet<String> attrValuesAlreadyMentioned,
            HashSet<String> attrValuesToBeMentioned, HashMap<String, HashSet<String>> availableAttributeActions,
            MeaningRepresentation MR) {
        TObjectDoubleHashMap<String> generalFeatures = new TObjectDoubleHashMap<>();
        HashMap<String, TObjectDoubleHashMap<String>> valueSpecificFeatures = new HashMap<>();
        if (availableAttributeActions.containsKey(predicate)) {
            availableAttributeActions.get(predicate).forEach((action) -> {
                valueSpecificFeatures.put(action, new TObjectDoubleHashMap<String>());
            });
        }
        ArrayList<String> mentionedAttrValues = new ArrayList<>();
        previousGeneratedAttrs.stream().filter((attrValue) -> (!attrValue.equals(Action.TOKEN_START)
                && !attrValue.equals(Action.TOKEN_END))).forEachOrdered((attrValue) -> {
            mentionedAttrValues.add(attrValue);
        });

        for (int j = 1; j <= 1; j++) {
            String previousAttrValue = "@@";
            if (mentionedAttrValues.size() - j >= 0) {
                previousAttrValue = mentionedAttrValues.get(mentionedAttrValues.size() - j).trim();
            }
            generalFeatures.put("feature_attrValue_" + j + "_" + previousAttrValue, 1.0);
        }
        //Word N-Grams
        String prevAttrValue = "@@";
        if (mentionedAttrValues.size() - 1 >= 0) {
            prevAttrValue = mentionedAttrValues.get(mentionedAttrValues.size() - 1).trim();
        }
        String prev2AttrValue = "@@";
        if (mentionedAttrValues.size() - 2 >= 0) {
            prev2AttrValue = mentionedAttrValues.get(mentionedAttrValues.size() - 2).trim();
        }
        String prev3AttrValue = "@@";
        if (mentionedAttrValues.size() - 3 >= 0) {
            prev3AttrValue = mentionedAttrValues.get(mentionedAttrValues.size() - 3).trim();
        }
        String prev4AttrValue = "@@";
        if (mentionedAttrValues.size() - 4 >= 0) {
            prev4AttrValue = mentionedAttrValues.get(mentionedAttrValues.size() - 4).trim();
        }
        String prev5AttrValue = "@@";
        if (mentionedAttrValues.size() - 5 >= 0) {
            prev5AttrValue = mentionedAttrValues.get(mentionedAttrValues.size() - 5).trim();
        }

        String prevBigramAttrValue = prev2AttrValue + "|" + prevAttrValue;
        String prevTrigramAttrValue = prev3AttrValue + "|" + prev2AttrValue + "|" + prevAttrValue;
        String prev4gramAttrValue = prev4AttrValue + "|" + prev3AttrValue + "|" + prev2AttrValue + "|" + prevAttrValue;
        String prev5gramAttrValue = prev5AttrValue + "|" + prev4AttrValue + "|" + prev3AttrValue + "|" + prev2AttrValue + "|" + prevAttrValue;
        generalFeatures.put("feature_attrValue_bigram_" + prevBigramAttrValue, 1.0);
        generalFeatures.put("feature_attrValue_trigram_" + prevTrigramAttrValue, 1.0);
        generalFeatures.put("feature_attrValue_4gram_" + prev4gramAttrValue, 1.0);
        generalFeatures.put("feature_attrValue_5gram_" + prev5gramAttrValue, 1.0);

        //If arguments have been generated or not
        for (int i = 0; i < mentionedAttrValues.size(); i++) {
            generalFeatures.put("feature_attrValue_allreadyMentioned_" + mentionedAttrValues.get(i), 1.0);
        }
        //If arguments should still be generated or not
        attrValuesToBeMentioned.forEach((attrValue) -> {
            generalFeatures.put("feature_attrValue_toBeMentioned_" + attrValue, 1.0);
        }); //Which attrs are in the MR and which are not

        if (availableAttributeActions.containsKey(predicate)) {
            availableAttributeActions.get(predicate).forEach((attribute) -> {
                if (MR.getAttributeValues().keySet().contains(attribute)) {
                    generalFeatures.put("feature_attr_inMR_" + attribute, 1.0);
                } else {
                    generalFeatures.put("feature_attr_notInMR_" + attribute, 1.0);
                }
            });
        }

        ArrayList<String> mentionedAttrs = new ArrayList<>();
        for (int i = 0; i < mentionedAttrValues.size(); i++) {
            String attr = mentionedAttrValues.get(i);
            if (attr.contains("=")) {
                attr = mentionedAttrValues.get(i).substring(0, mentionedAttrValues.get(i).indexOf('='));
            }
            mentionedAttrs.add(attr);
        }
        HashSet<String> attrsToBeMentioned = new HashSet<>();
        attrValuesToBeMentioned.stream().map((attrValue) -> {
            String attr = attrValue;
            if (attr.contains("=")) {
                attr = attrValue.substring(0, attrValue.indexOf('='));
            }
            return attr;
        }).forEachOrdered((attr) -> {
            attrsToBeMentioned.add(attr);
        });

        for (int j = 1; j <= 1; j++) {
            String previousAttr = "";
            if (mentionedAttrs.size() - j >= 0) {
                previousAttr = mentionedAttrs.get(mentionedAttrs.size() - j).trim();
            }
            if (!previousAttr.isEmpty()) {
                generalFeatures.put("feature_attr_" + j + "_" + previousAttr, 1.0);
            } else {
                generalFeatures.put("feature_attr_" + j + "_@@", 1.0);
            }
        }
        //Word N-Grams
        String prevAttr = "@@";
        if (mentionedAttrs.size() - 1 >= 0) {
            prevAttr = mentionedAttrs.get(mentionedAttrs.size() - 1).trim();
        }
        String prev2Attr = "@@";
        if (mentionedAttrs.size() - 2 >= 0) {
            prev2Attr = mentionedAttrs.get(mentionedAttrs.size() - 2).trim();
        }
        String prev3Attr = "@@";
        if (mentionedAttrs.size() - 3 >= 0) {
            prev3Attr = mentionedAttrs.get(mentionedAttrs.size() - 3).trim();
        }
        String prev4Attr = "@@";
        if (mentionedAttrs.size() - 4 >= 0) {
            prev4Attr = mentionedAttrs.get(mentionedAttrs.size() - 4).trim();
        }
        String prev5Attr = "@@";
        if (mentionedAttrs.size() - 5 >= 0) {
            prev5Attr = mentionedAttrs.get(mentionedAttrs.size() - 5).trim();
        }

        String prevBigramAttr = prev2Attr + "|" + prevAttr;
        String prevTrigramAttr = prev3Attr + "|" + prev2Attr + "|" + prevAttr;
        String prev4gramAttr = prev4Attr + "|" + prev3Attr + "|" + prev2Attr + "|" + prevAttr;
        String prev5gramAttr = prev5Attr + "|" + prev4Attr + "|" + prev3Attr + "|" + prev2Attr + "|" + prevAttr;

        generalFeatures.put("feature_attr_bigram_" + prevBigramAttr, 1.0);
        generalFeatures.put("feature_attr_trigram_" + prevTrigramAttr, 1.0);
        generalFeatures.put("feature_attr_4gram_" + prev4gramAttr, 1.0);
        generalFeatures.put("feature_attr_5gram_" + prev5gramAttr, 1.0);

        //If arguments have been generated or not
        attrValuesAlreadyMentioned.forEach((attr) -> {
            generalFeatures.put("feature_attr_alreadyMentioned_" + attr, 1.0);
        });
        //If arguments should still be generated or not
        attrsToBeMentioned.forEach((attr) -> {
            generalFeatures.put("feature_attr_toBeMentioned_" + attr, 1.0);
        });

        //Attr specific features (and global features)
        if (availableAttributeActions.containsKey(predicate)) {
            //No need to go through all actions, just actions present in the MR
            HashSet<String> relevantActions = new HashSet<String>();
            for (String attr : MR.getAttributeValues().keySet()) {
                if (availableAttributeActions.containsKey(cleanAndGetAttr(attr))) {
                    relevantActions.add(cleanAndGetAttr(attr));
                }
            }

            for (String action : relevantActions) {
                //for (String action : availableAttributeActions.get(predicate)) {
                if (action.equals(Action.TOKEN_END)) {
                    if (attrsToBeMentioned.isEmpty()) {
                        valueSpecificFeatures.get(action).put("global_feature_specific_allAttrValuesMentioned", 1.0);
                    } else {
                        valueSpecificFeatures.get(action).put("global_feature_specific_allAttrValuesNotMentioned", 1.0);
                    }
                } else {
                    //Is attr in MR?
                    if (MR.getAttributeValues().get(action) != null) {
                        valueSpecificFeatures.get(action).put("global_feature_specific_isInMR", 1.0);
                    } else {
                        valueSpecificFeatures.get(action).put("global_feature_specific_isNotInMR", 1.0);
                    }
                    //Is attr already mentioned right before
                    if (prevAttr.equals(action)) {
                        valueSpecificFeatures.get(action).put("global_feature_specific_attrFollowingSameAttr", 1.0);
                    } else {
                        valueSpecificFeatures.get(action).put("global_feature_specific_attrNotFollowingSameAttr", 1.0);
                    }
                    //Is attr already mentioned
                    attrValuesAlreadyMentioned.stream().map((attrValue) -> {
                        if (attrValue.indexOf('=') == -1) {
                        }
                        return attrValue;
                    }).filter((attrValue) -> (attrValue.substring(0, attrValue.indexOf('=')).equals(action))).forEachOrdered((_item) -> {
                        valueSpecificFeatures.get(action).put("global_feature_specific_attrAlreadyMentioned", 1.0);
                    });
                    //Is attr to be mentioned (has value to express)
                    boolean toBeMentioned = false;
                    for (String attrValue : attrValuesToBeMentioned) {
                        if (attrValue.substring(0, attrValue.indexOf('=')).equals(action)) {
                            toBeMentioned = true;
                            valueSpecificFeatures.get(action).put("global_feature_specific_attrToBeMentioned", 1.0);
                        }
                    }
                    if (!toBeMentioned) {
                        valueSpecificFeatures.get(action).put("global_feature_specific_attrNotToBeMentioned", 1.0);
                    }
                }
                HashSet<String> keys = new HashSet<>(valueSpecificFeatures.get(action).keySet());
                keys.forEach((feature1) -> {
                    keys.stream().filter((feature2) -> (valueSpecificFeatures.get(action).get(feature1) == 1.0
                            && valueSpecificFeatures.get(action).get(feature2) == 1.0
                            && feature1.compareTo(feature2) < 0)).forEachOrdered((feature2) -> {
                        valueSpecificFeatures.get(action).put(feature1 + "&&" + feature2, 1.0);
                    });
                });

                String nextValue = chooseNextValue(action, attrValuesToBeMentioned);
                if (nextValue.isEmpty() && !action.equals(Action.TOKEN_END)) {
                    valueSpecificFeatures.get(action).put("global_feature_LMAttr_score", 0.0);
                } else {
                    ArrayList<String> fullGramLM = new ArrayList<>();
                    for (int i = 0; i < mentionedAttrValues.size(); i++) {
                        fullGramLM.add(mentionedAttrValues.get(i));
                    }
                    ArrayList<String> prev5attrValueGramLM = new ArrayList<>();
                    int j = 0;
                    for (int i = mentionedAttrValues.size() - 1; (i >= 0 && j < 5); i--) {
                        prev5attrValueGramLM.add(0, mentionedAttrValues.get(i));
                        j++;
                    }
                    if (!action.equals(Action.TOKEN_END)) {
                        prev5attrValueGramLM.add(action + "=" + chooseNextValue(action, attrValuesToBeMentioned));
                    } else {
                        prev5attrValueGramLM.add(action);
                    }
                    while (prev5attrValueGramLM.size() < 4) {
                        prev5attrValueGramLM.add(0, "@@");
                    }

                    double afterLMScore = getContentLMsPerPredicate().get(predicate).getProbability(prev5attrValueGramLM);
                    valueSpecificFeatures.get(action).put("global_feature_LMAttr_score", afterLMScore);

                    afterLMScore = getContentLMsPerPredicate().get(predicate).getProbability(fullGramLM);
                    valueSpecificFeatures.get(action).put("global_feature_LMAttrFull_score", afterLMScore);
                }
            }
        }
        return new Instance(generalFeatures, valueSpecificFeatures, costs);

    }

    @Override
    public Instance createWordInstance(String predicate, Action bestAction,
            ArrayList<String> previousGeneratedAttributes, ArrayList<Action> previousGeneratedWords,
            ArrayList<String> nextGeneratedAttributes, HashSet<String> attrValuesAlreadyMentioned,
            HashSet<String> attrValuesThatFollow, boolean wasValueMentioned,
            HashMap<String, HashSet<Action>> availableWordActions,
            MeaningRepresentation MR) {
        TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();

        String attr = bestAction.getAttribute();
        if (bestAction.getAttribute().contains("=")) {
            attr = bestAction.getAttribute().substring(0, bestAction.getAttribute().indexOf('='));
        }
        if (!availableWordActions.containsKey(attr)) {
            System.out.println(attr);
            System.out.println("doesn't contain attr " + attr);
            System.exit(0);
        }

        if (!bestAction.getWord().trim().isEmpty()) {
            //COSTS

            for (Action action : availableWordActions.get(attr)) {
                if (action.getWord().equalsIgnoreCase(bestAction.getWord().trim())) {
                    costs.put(action.getAction(), 0.0);
                } else {
                    costs.put(action.getAction(), 1.0);
                }
            }

            if (bestAction.getWord().trim().equalsIgnoreCase(Action.TOKEN_END)) {
                costs.put(Action.TOKEN_END, 0.0);
            } else {
                costs.put(Action.TOKEN_END, 1.0);
            }
        }

        return createWordInstanceWithCosts(predicate, bestAction.getAttribute(), costs,
                previousGeneratedAttributes, previousGeneratedWords, nextGeneratedAttributes,
                attrValuesAlreadyMentioned, attrValuesThatFollow, wasValueMentioned, availableWordActions, MR);
    }

    @Override
    public Instance createWordInstanceWithCosts(String predicate, String currentAttrValue,
            TObjectDoubleHashMap<String> costs, ArrayList<String> generatedAttributes,
            ArrayList<Action> previousGeneratedWords, ArrayList<String> nextGeneratedAttributes,
            HashSet<String> attrValuesAlreadyMentioned, HashSet<String> attrValuesThatFollow, boolean wasValueMentioned,
            HashMap<String, HashSet<Action>> availableWordActions,
            MeaningRepresentation MR) {
        String currentAttr = currentAttrValue;
        String currentValue = "";
        if (currentAttr.contains("=")) {
            currentAttr = currentAttrValue.substring(0, currentAttrValue.indexOf('='));
            currentValue = currentAttrValue.substring(currentAttrValue.indexOf('=') + 1);
        }
        TObjectDoubleHashMap<String> generalFeatures = new TObjectDoubleHashMap<>();

        ArrayList<Action> generatedWords = new ArrayList<>();
        ArrayList<Action> generatedWordsInSameAttrValue = new ArrayList<>();
        ArrayList<String> generatedPhrase = new ArrayList<>();
        for (int i = 0; i < previousGeneratedWords.size(); i++) {
            Action a = previousGeneratedWords.get(i);
            if (!a.getWord().equals(Action.TOKEN_START)
                    && !a.getWord().equals(Action.TOKEN_END)) {
                generatedWords.add(a);
                generatedPhrase.add(a.getWord());
                if (a.getAttribute().equals(currentAttrValue)) {
                    generatedWordsInSameAttrValue.add(a);
                }
            }
        }
        //Previous word features
        for (int j = 1; j <= 1; j++) {
            String previousWord = "@@";
            if (generatedWords.size() - j >= 0) {
                previousWord = generatedWords.get(generatedWords.size() - j).getWord().trim();
            }
            generalFeatures.put("feature_word_" + j + "_" + previousWord.toLowerCase(), 1.0);
        }
        String prevWord = "@@";
        if (generatedWords.size() - 1 >= 0) {
            prevWord = generatedWords.get(generatedWords.size() - 1).getWord().trim();
        }
        String prev2Word = "@@";
        if (generatedWords.size() - 2 >= 0) {
            prev2Word = generatedWords.get(generatedWords.size() - 2).getWord().trim();
        }
        String prev3Word = "@@";
        if (generatedWords.size() - 3 >= 0) {
            prev3Word = generatedWords.get(generatedWords.size() - 3).getWord().trim();
        }
        /*
        String prev4Word = "@@";
        if (generatedWords.size() - 4 >= 0) {
            prev4Word = generatedWords.get(generatedWords.size() - 4).getWord().trim();
        }
        String prev5Word = "@@";
        if (generatedWords.size() - 5 >= 0) {
            prev5Word = generatedWords.get(generatedWords.size() - 5).getWord().trim();
        }*/

        String prevBigram = prev2Word + "|" + prevWord;
        String prevTrigram = prev3Word + "|" + prev2Word + "|" + prevWord;
        //String prev4gram = prev4Word + "|" + prev3Word + "|" + prev2Word + "|" + prevWord;
        //String prev5gram = prev5Word + "|" + prev4Word + "|" + prev3Word + "|" + prev2Word + "|" + prevWord;

        generalFeatures.put("feature_word_bigram_" + prevBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_word_trigram_" + prevTrigram.toLowerCase(), 1.0);
        //generalFeatures.put("feature_word_4gram_" + prev4gram.toLowerCase(), 1.0);
        //generalFeatures.put("feature_word_5gram_" + prev5gram.toLowerCase(), 1.0);

        //Previous Attr|Word features
        for (int j = 1; j <= 1; j++) {
            String previousAttrWord = "@@";
            if (generatedWords.size() - j >= 0) {
                if (generatedWords.get(generatedWords.size() - j).getAttribute().contains("=")) {
                    previousAttrWord = generatedWords.get(generatedWords.size() - j).getAttribute().trim().substring(0, generatedWords.get(generatedWords.size() - j).getAttribute().indexOf('=')) + "|" + generatedWords.get(generatedWords.size() - j).getWord().trim();
                } else {
                    previousAttrWord = generatedWords.get(generatedWords.size() - j).getAttribute().trim() + "|" + generatedWords.get(generatedWords.size() - j).getWord().trim();
                }
            }
            generalFeatures.put("feature_attrWord_" + j + "_" + previousAttrWord.toLowerCase(), 1.0);
        }
        String prevAttrWord = "@@";
        if (generatedWords.size() - 1 >= 0) {
            if (generatedWords.get(generatedWords.size() - 1).getAttribute().contains("=")) {
                prevAttrWord = generatedWords.get(generatedWords.size() - 1).getAttribute().trim().substring(0, generatedWords.get(generatedWords.size() - 1).getAttribute().indexOf('=')) + ":" + generatedWords.get(generatedWords.size() - 1).getWord().trim();
            } else {
                prevAttrWord = generatedWords.get(generatedWords.size() - 1).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 1).getWord().trim();

            }
        }
        String prev2AttrWord = "@@";
        if (generatedWords.size() - 2 >= 0) {
            if (generatedWords.get(generatedWords.size() - 2).getAttribute().contains("=")) {
                prev2AttrWord = generatedWords.get(generatedWords.size() - 2).getAttribute().trim().substring(0, generatedWords.get(generatedWords.size() - 2).getAttribute().indexOf('=')) + ":" + generatedWords.get(generatedWords.size() - 2).getWord().trim();
            } else {
                prev2AttrWord = generatedWords.get(generatedWords.size() - 2).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 2).getWord().trim();
            }
        }
        String prev3AttrWord = "@@";
        if (generatedWords.size() - 3 >= 0) {
            if (generatedWords.get(generatedWords.size() - 3).getAttribute().contains("=")) {
                prev3AttrWord = generatedWords.get(generatedWords.size() - 3).getAttribute().trim().substring(0, generatedWords.get(generatedWords.size() - 3).getAttribute().indexOf('=')) + ":" + generatedWords.get(generatedWords.size() - 3).getWord().trim();
            } else {
                prev3AttrWord = generatedWords.get(generatedWords.size() - 3).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 3).getWord().trim();
            }
        }/*
        String prev4AttrWord = "@@";
        if (generatedWords.size() - 4 >= 0) {
            if (generatedWords.get(generatedWords.size() - 4).getAttribute().contains("=")) {
                prev4AttrWord = generatedWords.get(generatedWords.size() - 4).getAttribute().trim().substring(0, generatedWords.get(generatedWords.size() - 4).getAttribute().indexOf('=')) + ":" + generatedWords.get(generatedWords.size() - 4).getWord().trim();
            } else {
                prev4AttrWord = generatedWords.get(generatedWords.size() - 4).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 4).getWord().trim();
            }
        }
        String prev5AttrWord = "@@";
        if (generatedWords.size() - 5 >= 0) {
            if (generatedWords.get(generatedWords.size() - 5).getAttribute().contains("=")) {
                prev5AttrWord = generatedWords.get(generatedWords.size() - 5).getAttribute().trim().substring(0, generatedWords.get(generatedWords.size() - 5).getAttribute().indexOf('=')) + ":" + generatedWords.get(generatedWords.size() - 5).getWord().trim();
            } else {
                prev5AttrWord = generatedWords.get(generatedWords.size() - 5).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 5).getWord().trim();
            }
        }*/

        String prevAttrWordBigram = prev2AttrWord + "|" + prevAttrWord;
        String prevAttrWordTrigram = prev3AttrWord + "|" + prev2AttrWord + "|" + prevAttrWord;
        //String prevAttrWord4gram = prev4AttrWord + "|" + prev3AttrWord + "|" + prev2AttrWord + "|" + prevAttrWord;
        //String prevAttrWord5gram = prev5AttrWord + "|" + prev4AttrWord + "|" + prev3AttrWord + "|" + prev2AttrWord + "|" + prevAttrWord;

        generalFeatures.put("feature_attrWord_bigram_" + prevAttrWordBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_attrWord_trigram_" + prevAttrWordTrigram.toLowerCase(), 1.0);
        //generalFeatures.put("feature_attrWord_4gram_" + prevAttrWord4gram.toLowerCase(), 1.0);
        //generalFeatures.put("feature_attrWord_5gram_" + prevAttrWord5gram.toLowerCase(), 1.0);

        //Previous AttrValue|Word features
        for (int j = 1; j <= 1; j++) {
            String previousAttrWord = "@@";
            if (generatedWords.size() - j >= 0) {
                previousAttrWord = generatedWords.get(generatedWords.size() - j).getAttribute().trim() + "|" + generatedWords.get(generatedWords.size() - j).getWord().trim();
            }
            generalFeatures.put("feature_attrValueWord_" + j + "_" + previousAttrWord.toLowerCase(), 1.0);
        }
        String prevAttrValueWord = "@@";
        if (generatedWords.size() - 1 >= 0) {
            prevAttrValueWord = generatedWords.get(generatedWords.size() - 1).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 1).getWord().trim();
        }
        String prev2AttrValueWord = "@@";
        if (generatedWords.size() - 2 >= 0) {
            prev2AttrValueWord = generatedWords.get(generatedWords.size() - 2).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 2).getWord().trim();
        }
        String prev3AttrValueWord = "@@";
        if (generatedWords.size() - 3 >= 0) {
            prev3AttrValueWord = generatedWords.get(generatedWords.size() - 3).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 3).getWord().trim();
        }/*
        String prev4AttrValueWord = "@@";
        if (generatedWords.size() - 4 >= 0) {
            prev4AttrValueWord = generatedWords.get(generatedWords.size() - 4).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 4).getWord().trim();
        }
        String prev5AttrValueWord = "@@";
        if (generatedWords.size() - 5 >= 0) {
            prev5AttrValueWord = generatedWords.get(generatedWords.size() - 5).getAttribute().trim() + ":" + generatedWords.get(generatedWords.size() - 5).getWord().trim();
        }*/

        String prevAttrValueWordBigram = prev2AttrValueWord + "|" + prevAttrValueWord;
        String prevAttrValueWordTrigram = prev3AttrValueWord + "|" + prev2AttrValueWord + "|" + prevAttrValueWord;
        //String prevAttrValueWord4gram = prev4AttrValueWord + "|" + prev3AttrValueWord + "|" + prev2AttrValueWord + "|" + prevAttrValueWord;
        //String prevAttrValueWord5gram = prev5AttrValueWord + "|" + prev4AttrValueWord + "|" + prev3AttrValueWord + "|" + prev2AttrValueWord + "|" + prevAttrValueWord;

        generalFeatures.put("feature_attrValueWord_bigram_" + prevAttrValueWordBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_attrValueWord_trigram_" + prevAttrValueWordTrigram.toLowerCase(), 1.0);
        //generalFeatures.put("feature_attrValueWord_4gram_" + prevAttrValueWord4gram.toLowerCase(), 1.0);
        //generalFeatures.put("feature_attrValueWord_5gram_" + prevAttrValueWord5gram.toLowerCase(), 1.0);

        //Previous attrValue features
        int attributeSize = generatedAttributes.size();
        for (int j = 1; j <= 1; j++) {
            String previousAttrValue = "@@";
            if (attributeSize - j >= 0) {
                previousAttrValue = generatedAttributes.get(attributeSize - j).trim();
            }
            generalFeatures.put("feature_attrValue_" + j + "_" + previousAttrValue, 1.0);
        }
        String prevAttrValue = "@@";
        if (attributeSize - 1 >= 0) {
            prevAttrValue = generatedAttributes.get(attributeSize - 1).trim();
        }
        String prev2AttrValue = "@@";
        if (attributeSize - 2 >= 0) {
            prev2AttrValue = generatedAttributes.get(attributeSize - 2).trim();
        }
        String prev3AttrValue = "@@";
        if (attributeSize - 3 >= 0) {
            prev3AttrValue = generatedAttributes.get(attributeSize - 3).trim();
        }/*
        String prev4AttrValue = "@@";
        if (attributeSize - 4 >= 0) {
            prev4AttrValue = generatedAttributes.get(attributeSize - 4).trim();
        }
        String prev5AttrValue = "@@";
        if (attributeSize - 5 >= 0) {
            prev5AttrValue = generatedAttributes.get(attributeSize - 5).trim();
        }*/

        String prevAttrBigramValue = prev2AttrValue + "|" + prevAttrValue;
        String prevAttrTrigramValue = prev3AttrValue + "|" + prev2AttrValue + "|" + prevAttrValue;
        //String prevAttr4gramValue = prev4AttrValue + "|" + prev3AttrValue + "|" + prev2AttrValue + "|" + prevAttrValue;
        //String prevAttr5gramValue = prev5AttrValue + "|" + prev4AttrValue + "|" + prev3AttrValue + "|" + prev2AttrValue + "|" + prevAttrValue;

        generalFeatures.put("feature_attrValue_bigram_" + prevAttrBigramValue.toLowerCase(), 1.0);
        generalFeatures.put("feature_attrValue_trigram_" + prevAttrTrigramValue.toLowerCase(), 1.0);
        //generalFeatures.put("feature_attrValue_4gram_" + prevAttr4gramValue.toLowerCase(), 1.0);
        //generalFeatures.put("feature_attrValue_5gram_" + prevAttr5gramValue.toLowerCase(), 1.0);

        //Previous attr features
        for (int j = 1; j <= 1; j++) {
            String previousAttr = "@@";
            if (attributeSize - j >= 0) {
                if (generatedAttributes.get(attributeSize - j).contains("=")) {
                    previousAttr = generatedAttributes.get(attributeSize - j).trim().substring(0, generatedAttributes.get(attributeSize - j).indexOf('='));
                } else {
                    previousAttr = generatedAttributes.get(attributeSize - j).trim();
                }
            }
            generalFeatures.put("feature_attr_" + j + "_" + previousAttr, 1.0);
        }
        String prevAttr = "@@";
        if (attributeSize - 1 >= 0) {
            if (generatedAttributes.get(attributeSize - 1).contains("=")) {
                prevAttr = generatedAttributes.get(attributeSize - 1).trim().substring(0, generatedAttributes.get(attributeSize - 1).indexOf('='));
            } else {
                prevAttr = generatedAttributes.get(attributeSize - 1).trim();
            }
        }
        String prev2Attr = "@@";
        if (attributeSize - 2 >= 0) {
            if (generatedAttributes.get(attributeSize - 2).contains("=")) {
                prev2Attr = generatedAttributes.get(attributeSize - 2).trim().substring(0, generatedAttributes.get(attributeSize - 2).indexOf('='));
            } else {
                prev2Attr = generatedAttributes.get(attributeSize - 2).trim();
            }
        }
        String prev3Attr = "@@";
        if (attributeSize - 3 >= 0) {
            if (generatedAttributes.get(attributeSize - 3).contains("=")) {
                prev3Attr = generatedAttributes.get(attributeSize - 3).trim().substring(0, generatedAttributes.get(attributeSize - 3).indexOf('='));
            } else {
                prev3Attr = generatedAttributes.get(attributeSize - 3).trim();
            }
        }/*
        String prev4Attr = "@@";
        if (attributeSize - 4 >= 0) {
            if (generatedAttributes.get(attributeSize - 4).contains("=")) {
                prev4Attr = generatedAttributes.get(attributeSize - 4).trim().substring(0, generatedAttributes.get(attributeSize - 4).indexOf('='));
            } else {
                prev4Attr = generatedAttributes.get(attributeSize - 4).trim();
            }
        }
        String prev5Attr = "@@";
        if (attributeSize - 5 >= 0) {
            if (generatedAttributes.get(attributeSize - 5).contains("=")) {
                prev5Attr = generatedAttributes.get(attributeSize - 5).trim().substring(0, generatedAttributes.get(attributeSize - 5).indexOf('='));
            } else {
                prev5Attr = generatedAttributes.get(attributeSize - 5).trim();
            }
        }*/

        String prevAttrBigram = prev2Attr + "|" + prevAttr;
        String prevAttrTrigram = prev3Attr + "|" + prev2Attr + "|" + prevAttr;
        //String prevAttr4gram = prev4Attr + "|" + prev3Attr + "|" + prev2Attr + "|" + prevAttr;
        //String prevAttr5gram = prev5Attr + "|" + prev4Attr + "|" + prev3Attr + "|" + prev2Attr + "|" + prevAttr;

        generalFeatures.put("feature_attr_bigram_" + prevAttrBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_attr_trigram_" + prevAttrTrigram.toLowerCase(), 1.0);
        //generalFeatures.put("feature_attr_4gram_" + prevAttr4gram.toLowerCase(), 1.0);
        //generalFeatures.put("feature_attr_5gram_" + prevAttr5gram.toLowerCase(), 1.0);

        //Next attr features
        for (int j = 0; j < 1; j++) {
            String nextAttr = "@@";
            if (j < nextGeneratedAttributes.size()) {
                if (nextGeneratedAttributes.get(j).contains("=")) {
                    nextAttr = nextGeneratedAttributes.get(j).trim().substring(0, nextGeneratedAttributes.get(j).indexOf('='));
                } else {
                    nextAttr = nextGeneratedAttributes.get(j).trim();
                }
            }
            generalFeatures.put("feature_nextAttr_" + j + "_" + nextAttr, 1.0);
        }
        String nextAttr = "@@";
        if (0 < nextGeneratedAttributes.size()) {
            if (nextGeneratedAttributes.get(0).contains("=")) {
                nextAttr = nextGeneratedAttributes.get(0).trim().substring(0, nextGeneratedAttributes.get(0).indexOf('='));
            } else {
                nextAttr = nextGeneratedAttributes.get(0).trim();
            }
        }
        String next2Attr = "@@";
        if (1 < nextGeneratedAttributes.size()) {
            if (nextGeneratedAttributes.get(1).contains("=")) {
                next2Attr = nextGeneratedAttributes.get(1).trim().substring(0, nextGeneratedAttributes.get(1).indexOf('='));
            } else {
                next2Attr = nextGeneratedAttributes.get(1).trim();
            }
        }
        String next3Attr = "@@";
        if (2 < nextGeneratedAttributes.size()) {
            if (nextGeneratedAttributes.get(2).contains("=")) {
                next3Attr = nextGeneratedAttributes.get(2).trim().substring(0, nextGeneratedAttributes.get(2).indexOf('='));
            } else {
                next3Attr = nextGeneratedAttributes.get(2).trim();
            }
        }/*
        String next4Attr = "@@";
        if (3 < nextGeneratedAttributes.size()) {
            if (nextGeneratedAttributes.get(3).contains("=")) {
                next4Attr = nextGeneratedAttributes.get(3).trim().substring(0, nextGeneratedAttributes.get(3).indexOf('='));
            } else {
                next4Attr = nextGeneratedAttributes.get(3).trim();
            }
        }
        String next5Attr = "@@";
        if (4 < nextGeneratedAttributes.size()) {
            if (nextGeneratedAttributes.get(4).contains("=")) {
                next5Attr = nextGeneratedAttributes.get(4).trim().substring(0, nextGeneratedAttributes.get(4).indexOf('='));
            } else {
                next5Attr = nextGeneratedAttributes.get(4).trim();
            }
        }*/

        String nextAttrBigram = nextAttr + "|" + next2Attr;
        String nextAttrTrigram = nextAttr + "|" + next2Attr + "|" + next3Attr;
        //String nextAttr4gram = nextAttr + "|" + next2Attr + "|" + next3Attr + "|" + next4Attr;
        //String nextAttr5gram = nextAttr + "|" + next2Attr + "|" + next3Attr + "|" + next4Attr + "|" + next5Attr;

        generalFeatures.put("feature_nextAttr_bigram_" + nextAttrBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_nextAttr_trigram_" + nextAttrTrigram.toLowerCase(), 1.0);
        //generalFeatures.put("feature_nextAttr_4gram_" + nextAttr4gram.toLowerCase(), 1.0);
        //generalFeatures.put("feature_nextAttr_5gram_" + nextAttr5gram.toLowerCase(), 1.0);

        //Next attrValue features
        for (int j = 0; j < 1; j++) {
            String nextAttrValue = "@@";
            if (j < nextGeneratedAttributes.size()) {
                nextAttrValue = nextGeneratedAttributes.get(j).trim();
            }
            generalFeatures.put("feature_nextAttrValue_" + j + "_" + nextAttrValue, 1.0);
        }
        String nextAttrValue = "@@";
        if (0 < nextGeneratedAttributes.size()) {
            nextAttrValue = nextGeneratedAttributes.get(0).trim();
        }
        String next2AttrValue = "@@";
        if (1 < nextGeneratedAttributes.size()) {
            next2AttrValue = nextGeneratedAttributes.get(1).trim();
        }
        String next3AttrValue = "@@";
        if (2 < nextGeneratedAttributes.size()) {
            next3AttrValue = nextGeneratedAttributes.get(2).trim();
        }/*
        String next4AttrValue = "@@";
        if (3 < nextGeneratedAttributes.size()) {
            next4AttrValue = nextGeneratedAttributes.get(3).trim();
        }
        String next5AttrValue = "@@";
        if (4 < nextGeneratedAttributes.size()) {
            next5AttrValue = nextGeneratedAttributes.get(4).trim();
        }*/

        String nextAttrValueBigram = nextAttrValue + "|" + next2AttrValue;
        String nextAttrValueTrigram = nextAttrValue + "|" + next2AttrValue + "|" + next3AttrValue;
        //String nextAttrValue4gram = nextAttrValue + "|" + next2AttrValue + "|" + next3AttrValue + "|" + next4AttrValue;
        //String nextAttrValue5gram = nextAttrValue + "|" + next2AttrValue + "|" + next3AttrValue + "|" + next4AttrValue + "|" + next5AttrValue;

        generalFeatures.put("feature_nextAttrValue_bigram_" + nextAttrValueBigram.toLowerCase(), 1.0);
        generalFeatures.put("feature_nextAttrValue_trigram_" + nextAttrValueTrigram.toLowerCase(), 1.0);
        //generalFeatures.put("feature_nextAttrValue_4gram_" + nextAttrValue4gram.toLowerCase(), 1.0);
        //generalFeatures.put("feature_nextAttrValue_5gram_" + nextAttrValue5gram.toLowerCase(), 1.0);

        //If values have already been generated or not
        generalFeatures.put("feature_valueToBeMentioned_" + currentValue.toLowerCase(), 1.0);
        if (wasValueMentioned) {
            generalFeatures.put("feature_wasValueMentioned_true", 1.0);
        } else {
            //generalFeatures.put("feature_wasValueMentioned_false", 1.0);
        }
        HashSet<String> valuesThatFollow = new HashSet<>();
        attrValuesThatFollow.stream().map((attrValue) -> {
            generalFeatures.put("feature_attrValuesThatFollow_" + attrValue.toLowerCase(), 1.0);
            return attrValue;
        }).forEachOrdered((attrValue) -> {
            if (attrValue.contains("=")) {
                String v = attrValue.substring(attrValue.indexOf('=') + 1);
                if (v.matches("[xX][0-9]+")) {
                    String attr = attrValue.substring(0, attrValue.indexOf('='));
                    valuesThatFollow.add(Action.TOKEN_X + attr + "_" + v.substring(1));
                } else {
                    valuesThatFollow.add(v);
                }
                generalFeatures.put("feature_attrsThatFollow_" + attrValue.substring(0, attrValue.indexOf('=')).toLowerCase(), 1.0);
            } else {
                generalFeatures.put("feature_attrsThatFollow_" + attrValue.toLowerCase(), 1.0);
            }
        });
        if (valuesThatFollow.isEmpty()) {
            generalFeatures.put("feature_noAttrsFollow", 1.0);
        } else {
            generalFeatures.put("feature_noAttrsFollow", 0.0);
        }
        HashSet<String> mentionedValues = new HashSet<>();
        attrValuesAlreadyMentioned.stream().map((attrValue) -> {
            generalFeatures.put("feature_attrValuesAlreadyMentioned_" + attrValue.toLowerCase(), 1.0);
            return attrValue;
        }).forEachOrdered((attrValue) -> {
            if (attrValue.contains("=")) {
                generalFeatures.put("feature_attrsAlreadyMentioned_" + attrValue.substring(0, attrValue.indexOf('=')).toLowerCase(), 1.0);
                String v = attrValue.substring(attrValue.indexOf('=') + 1);
                if (v.matches("[xX][0-9]+")) {
                    String attr = attrValue.substring(0, attrValue.indexOf('='));
                    mentionedValues.add(Action.TOKEN_X + attr + "_" + v.substring(1));
                } else {
                    mentionedValues.add(v);
                }
            } else {
                generalFeatures.put("feature_attrsAlreadyMentioned_" + attrValue.toLowerCase(), 1.0);
            }
        });
        //Word specific features (and also global features)

        HashMap<String, TObjectDoubleHashMap<String>> valueSpecificFeatures = new HashMap<>();
        for (Action action : availableWordActions.get(currentAttr)) {
            valueSpecificFeatures.put(action.getAction(), new TObjectDoubleHashMap<String>());
        }
        for (Action action : availableWordActions.get(currentAttr)) {

            //Is word same as previous word
            if (prevWord.equals(action.getWord())) {
                //valueSpecificFeatures.get(action.getAction()).put("feature_specific_sameAsPreviousWord", 1.0);
                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_sameAsPreviousWord", 1.0);
            } else {
                //valueSpecificFeatures.get(action.getAction()).put("feature_specific_notSameAsPreviousWord", 1.0);
                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_notSameAsPreviousWord", 1.0);
            }
            //Has word appeared in the same attrValue before
            generatedWords.forEach((previousAction) -> {
                if (previousAction.getWord().equals(action.getWord())
                        && previousAction.getAttribute().equals(currentAttrValue)) {
                    //valueSpecificFeatures.get(action.getAction()).put("feature_specific_appearedInSameAttrValue", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_appearedInSameAttrValue", 1.0);
                } else {
                    //valueSpecificFeatures.get(action.getAction()).put("feature_specific_notAppearedInSameAttrValue", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_notAppearedInSameAttrValue", 1.0);
                }
            });
            //Has word appeared before
            generatedWords.forEach((previousAction) -> {
                if (previousAction.getWord().equals(action.getWord())) {
                    //valueSpecificFeatures.get(action.getAction()).put("feature_specific_appeared", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_appeared", 1.0);
                } else {
                    //valueSpecificFeatures.get(action.getAction()).put("feature_specific_notAppeared", 1.0);
                    //valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_notAppeared", 1.0);
                }
            });
            if (currentValue.equals("no")
                    || currentValue.equals("yes")
                    || currentValue.equals("yes or no")
                    || currentValue.equals("none")
                    || currentValue.equals("empty") //|| currentValue.equals("dont_care")
                    ) {
                //valueSpecificFeatures.get(action.getAction()).put("feature_specific_emptyValue", 1.0);
                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_emptyValue", 1.0);
            } else {
                //valueSpecificFeatures.get(action.getAction()).put("feature_specific_notEmptyValue", 1.0);
                //valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_notEmptyValue", 1.0);
            }

            HashSet<String> keys = new HashSet<>(valueSpecificFeatures.get(action.getAction()).keySet());
            keys.forEach((feature1) -> {
                keys.stream().filter((feature2) -> (valueSpecificFeatures.get(action.getAction()).get(feature1) == 1.0
                        && valueSpecificFeatures.get(action.getAction()).get(feature2) == 1.0
                        && feature1.compareTo(feature2) < 0)).forEachOrdered((feature2) -> {
                    valueSpecificFeatures.get(action.getAction()).put(feature1 + "&&" + feature2, 1.0);
                });
            });

            if (!action.getWord().startsWith(Action.TOKEN_X)
                    && !currentValue.equals("no")
                    && !currentValue.equals("yes")
                    && !currentValue.equals("yes or no")
                    && !currentValue.equals("none")
                    && !currentValue.equals("empty") //&& !currentValue.equals("dont_care")
                    ) {
                // CHANGED TO NOT REGARD OUT-OF-MR VALUES
                for (String value : getValueAlignments().keySet()) {
                    if (currentValue.equals(value)) {
                        for (ArrayList<String> alignedStr : getValueAlignments().get(value).keySet()) {
                            if (alignedStr.get(0).equals(action.getWord())) {
                                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_beginsValue_current", 1.0);
                            } else {
                                for (int i = 1; i < alignedStr.size(); i++) {
                                    if (alignedStr.get(i).equals(action.getWord())) {
                                        if (endsWith(generatedPhrase, new ArrayList<String>(alignedStr.subList(0, i + 1)))) {
                                            valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_inValue_current", 1.0);
                                        } else {
                                            valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_outOfValue", 1.0);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (action.getWord().equals(Action.TOKEN_END)) {
                    if (generatedWordsInSameAttrValue.isEmpty()) {
                        valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_closingEmptyAttr", 1.0);
                    }
                    if (!wasValueMentioned) {
                        valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_closingAttrWithValueNotMentioned", 1.0);
                    }

                    if (!prevWord.equals("@@")) {
                        boolean alignmentIsOpen = false;
                        for (String value : getValueAlignments().keySet()) {
                            if (currentValue.equals(value)) {
                                for (ArrayList<String> alignedStr : getValueAlignments().get(value).keySet()) {
                                    for (int i = 0; i < alignedStr.size() - 1; i++) {
                                        if (alignedStr.get(i).equals(prevWord)
                                                && endsWith(generatedPhrase, new ArrayList<>(alignedStr.subList(0, i + 1)))) {
                                            alignmentIsOpen = true;
                                        }
                                    }
                                }
                            }
                        }
                        if (alignmentIsOpen) {
                            valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_closingAttrWhileValueIsNotConcluded", 1.0);
                        }
                    }
                }
            } else if (currentValue.equals("no")
                    || currentValue.equals("yes")
                    || currentValue.equals("yes or no")
                    || currentValue.equals("none")
                    || currentValue.equals("empty") //|| currentValue.equals("dont_care")
                    ) {
                valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_XValue_notInMR", 1.0);
            } else {
                String currentValueVariant = "";
                if (currentValue.matches("[xX][0-9]+")) {
                    currentValueVariant = Action.TOKEN_X + currentAttr + "_" + currentValue.substring(1);
                }

                if (mentionedValues.contains(action.getWord())) {
                    //valueSpecificFeatures.get(action.getAction()).put("feature_specific_XValue_alreadyMentioned", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_XValue_alreadyMentioned", 1.0);
                } else if (currentValueVariant.equals(action.getWord())
                        && !currentValueVariant.isEmpty()) {
                    //valueSpecificFeatures.get(action.getAction()).put("feature_specific_XValue_current", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_XValue_current", 1.0);

                } else if (valuesThatFollow.contains(action.getWord())) {
                    //valueSpecificFeatures.get(action.getAction()).put("feature_specific_XValue_thatFollows", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_XValue_thatFollows", 1.0);
                } else {
                    //valueSpecificFeatures.get(action.getAction()).put("feature_specific_XValue_notInMR", 1.0);
                    valueSpecificFeatures.get(action.getAction()).put("global_feature_specific_XValue_notInMR", 1.0);
                }
            }

            //valueSpecificFeatures.get(action.getAction()).put("global_feature_abstractMR_" + mr.getAbstractMR(), 1.0);
            valueSpecificFeatures.get(action.getAction()).put("global_feature_currentValue_" + currentValue.toLowerCase(), 1.0);

            ArrayList<String> fullGramLM = new ArrayList<>();
            for (int i = 0; i < generatedWords.size(); i++) {
                fullGramLM.add(generatedWords.get(i).getWord());
            }

            ArrayList<String> prev5wordGramLM = new ArrayList<>();
            int j = 0;
            for (int i = generatedWords.size() - 1; (i >= 0 && j < 5); i--) {
                prev5wordGramLM.add(0, generatedWords.get(i).getWord());
                j++;
            }
            prev5wordGramLM.add(action.getWord());
            while (prev5wordGramLM.size() < 4) {
                prev5wordGramLM.add(0, "@@");
            }

            double afterLMScorePerPred5Gram = getWordLMsPerPredicate().get(predicate).getProbability(prev5wordGramLM);
            valueSpecificFeatures.get(action.getAction()).put("global_feature_LMWord_perPredicate_5gram_score", afterLMScorePerPred5Gram);

            double afterLMScorePerPred = getWordLMsPerPredicate().get(predicate).getProbability(fullGramLM);
            valueSpecificFeatures.get(action.getAction()).put("global_feature_LMWord_perPredicate_score", afterLMScorePerPred);
        }

        return new Instance(generalFeatures, valueSpecificFeatures, costs);
    }

    @Override
    public String postProcessWordSequence(MeaningRepresentation mr, ArrayList<Action> wordSequence) {

        HashSet<ArrayList<Action>> matched = new HashSet<>();
        ArrayList<Action> processedWordSequence = new ArrayList<>();
        wordSequence.forEach((act) -> {
            processedWordSequence.add(new Action(act));
        });
        if (!processedWordSequence.isEmpty()
                && processedWordSequence.get(processedWordSequence.size() - 1).getWord().equals(Action.TOKEN_END)
                && processedWordSequence.get(processedWordSequence.size() - 1).getAttribute().equals(Action.TOKEN_END)) {
            processedWordSequence.remove(processedWordSequence.size() - 1);
        }
        // REPLACE @unk@ TOKENS WITH MOST PROBABLY SEQ ACCORDING TO LANGUAGE MODELS
        for (int i = 0; i < processedWordSequence.size(); i++) {
            if (processedWordSequence.get(i).getWord().equals("@unk@")) {
                ArrayList<String> previousWords = new ArrayList<>();
                while (previousWords.size() < 4) {
                    previousWords.add(0, "@@");
                }
                for (int j = 0; j < i; j++) {
                    previousWords.add(processedWordSequence.get(j).getWord());
                }
                ArrayList<String> nextWords = new ArrayList<>();
                int j = i + 1;
                while (j < processedWordSequence.size() && !processedWordSequence.get(j).getWord().equals("@unk@")) {
                    nextWords.add(processedWordSequence.get(j).getWord());
                    j++;
                }

                HashMap<String, Double> availWordLMScore = new HashMap<>();
                for (Action word : getAvailableWordActions().get(mr.getPredicate()).get(cleanAndGetAttr(processedWordSequence.get(i).getAttribute()))) {
                    ArrayList<String> seq = new ArrayList<>(previousWords);
                    seq.add(word.getWord());
                    seq.addAll(nextWords);
                    double LMScore = getWordLMsPerPredicate().get(mr.getPredicate()).getProbability(seq);
                    availWordLMScore.put(word.getWord(), LMScore);
                }
                double maxLMScore = Double.MIN_VALUE;
                String maxWord = "";
                for (String word : availWordLMScore.keySet()) {
                    if (availWordLMScore.get(word) > maxLMScore) {
                        maxLMScore = availWordLMScore.get(word);
                        maxWord = word;
                    }
                }
                processedWordSequence.get(i).setWord(maxWord);
            }
        }
        if (getPunctuationPatterns().containsKey(mr.getPredicate())) {
            getPunctuationPatterns().get(mr.getPredicate()).keySet().forEach((surrounds) -> {
                int beforeNulls = 0;
                if (surrounds.get(0) == null) {
                    beforeNulls++;
                }
                if (surrounds.get(1) == null) {
                    beforeNulls++;
                }
                for (int i = 0 - beforeNulls; i < processedWordSequence.size(); i++) {
                    boolean matches = true;
                    int m = 0;
                    for (int s = 0; s < surrounds.size(); s++) {
                        if (surrounds.get(s) != null) {
                            if (i + s < processedWordSequence.size()) {
                                if (!processedWordSequence.get(i + s).getWord().equals(surrounds.get(s).getWord()) /*|| !cleanActionList.get(i).getAttribute().equals(surrounds.get(s).getAttribute())*/) {
                                    matches = false;
                                    s = surrounds.size();
                                } else {
                                    m++;
                                }
                            } else {
                                matches = false;
                                s = surrounds.size();
                            }
                        } else if (s < 2 && i + s >= 0) {
                            matches = false;
                            s = surrounds.size();
                        } else if (s >= 2 && i + s < processedWordSequence.size()) {
                            matches = false;
                            s = surrounds.size();
                        }
                    }
                    if (matches && m > 0) {
                        matched.add(surrounds);
                        processedWordSequence.add(i + 2, getPunctuationPatterns().get(mr.getPredicate()).get(surrounds));
                    }
                }
            });
        }
        boolean isLastPunct = true;
        if (processedWordSequence.contains(new Action("and", ""))) {
            for (int i = processedWordSequence.size() - 1; i > 0; i--) {
                if (processedWordSequence.get(i).getWord().equals(",")
                        && isLastPunct) {
                    isLastPunct = false;
                    processedWordSequence.get(i).setWord("and");
                } else if (processedWordSequence.get(i).getWord().equals("and")
                        && isLastPunct) {
                    isLastPunct = false;
                } else if (processedWordSequence.get(i).getWord().equals("and")
                        && !isLastPunct) {
                    processedWordSequence.get(i).setWord(",");
                }
            }
        }

        ArrayList<Action> cleanActionList = new ArrayList<>();
        processedWordSequence.stream().filter((action) -> (!action.getWord().equals(Action.TOKEN_START)
                && !action.getWord().equals(Action.TOKEN_END))).forEachOrdered((action) -> {
            cleanActionList.add(action);
        });

        String predictedWordSequence = " ";
        boolean previousIsTokenX = false;
        for (Action action : cleanActionList) {
            if (action.getWord().startsWith(Action.TOKEN_X)) {
                predictedWordSequence += " " + mr.getDelexicalizationMap().get(action.getWord());
                previousIsTokenX = true;
            } else {
                if (action.getWord().equals("-ly") && previousIsTokenX) {
                    predictedWordSequence += "ly";
                } else if (action.getWord().equals("s") && previousIsTokenX) {
                    predictedWordSequence += action.getWord();
                } else {
                    predictedWordSequence += " " + action.getWord();
                }
                previousIsTokenX = false;
            }
        }

        predictedWordSequence = predictedWordSequence.trim();
        if (mr.getPredicate().startsWith("?")
                && !predictedWordSequence.endsWith("?")) {
            if (predictedWordSequence.endsWith(".")) {
                predictedWordSequence = predictedWordSequence.substring(0, predictedWordSequence.length() - 1);
            }
            predictedWordSequence = predictedWordSequence.trim() + "?";
        } else if (!predictedWordSequence.endsWith(".") && !predictedWordSequence.endsWith("?")) {
            /*if (predictedWordSequence.endsWith("?")) {
                predictedWordSequence = predictedWordSequence.substring(0, predictedWordSequence.length() - 1);
            }*/
            predictedWordSequence = predictedWordSequence.trim() + ".";
        }
        predictedWordSequence = predictedWordSequence.replaceAll(" the the ", " the ").replaceAll("\\?", " \\? ").replaceAll(":", " : ").replaceAll("\\.", " \\. ").replaceAll(",", " , ").replaceAll("  ", " ").trim();
        predictedWordSequence = predictedWordSequence.replaceAll(" , \\. ", " \\. ").replaceAll(" and \\. ", " \\. ").replaceAll(" , \\? ", " \\? ").replaceAll(" and \\? ", " \\? ").replaceAll(" ,\\. ", " \\. ").replaceAll(" and\\. ", " \\. ").replaceAll(" ,\\? ", " \\? ").replaceAll(" and\\? ", " \\? ").trim();
        /*for (String comp : sillyCompositeWordsInData.keySet()) {
            predictedWordSequence = predictedWordSequence.replaceAll(comp, sillyCompositeWordsInData.get(comp));
        }*/
        if (predictedWordSequence.startsWith(",")
                || predictedWordSequence.startsWith(".")
                || predictedWordSequence.startsWith("?")) {
            predictedWordSequence = predictedWordSequence.substring(1).trim();
        }
        if (predictedWordSequence.startsWith(",")) {
            System.out.println(wordSequence);
            System.out.println(matched);
        }
        return predictedWordSequence;
    }

    public ArrayList<String> getPredictedAttrList(ArrayList<Action> wordSequence) {
        ArrayList<Action> cleanActionList = new ArrayList<>();
        wordSequence.stream().filter((action) -> (!action.getWord().equals(Action.TOKEN_START)
                && !action.getWord().equals(Action.TOKEN_END))).forEachOrdered((action) -> {
            cleanActionList.add(action);
        });

        ArrayList<String> predictedAttrList = new ArrayList<>();
        cleanActionList.forEach((action) -> {
            if (predictedAttrList.isEmpty()) {
                predictedAttrList.add(action.getAttribute());
            } else if (!predictedAttrList.get(predictedAttrList.size() - 1).equals(action.getAttribute())) {
                predictedAttrList.add(action.getAttribute());
            }
        });
        return predictedAttrList;
    }

    @Override
    public String postProcessRef(MeaningRepresentation mr, ArrayList<Action> refSeq) {
        String cleanedWords = "";
        for (Action nlWord : refSeq) {
            if (!nlWord.equals(new Action(Action.TOKEN_END, ""))
                    && !nlWord.equals(new Action(Action.TOKEN_START, ""))
                    && !nlWord.getWord().equals(Action.TOKEN_PUNCT)) {
                if (nlWord.getWord().startsWith(Action.TOKEN_X)) {
                    cleanedWords += " " + mr.getDelexicalizationMap().get(nlWord.getWord());
                } else {
                    cleanedWords += " " + nlWord.getWord();
                }
            }
        }
        if (!cleanedWords.trim().endsWith(".")) {
            cleanedWords += " .";
        }

        return cleanedWords.trim();
    }

    @Override
    public boolean loadInitClassifiers(int dataSize, HashMap<String, JAROW> trainedAttrClassifiers_0,
            HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_0) {

        String file1 = "cache/attrInitClassifiers" + "_" + dataSize;
        String file2 = "cache/wordInitClassifiers" + "_" + dataSize;
        FileInputStream fin1 = null;
        ObjectInputStream ois1 = null;
        FileInputStream fin2 = null;
        ObjectInputStream ois2 = null;
        if ((new File(file1)).exists()
                && (new File(file2)).exists()) {
            try {
                System.out.print("Load initial classifiers...");
                fin1 = new FileInputStream(file1);
                ois1 = new ObjectInputStream(fin1);
                Object o1 = ois1.readObject();
                if (o1 instanceof HashMap) {
                    trainedAttrClassifiers_0.putAll((Map<? extends String, ? extends JAROW>) o1);
                }

                fin2 = new FileInputStream(file2);
                ois2 = new ObjectInputStream(fin2);
                Object o2 = ois2.readObject();
                if (o2 instanceof HashMap) {
                    trainedWordClassifiers_0.putAll((Map<? extends String, ? extends HashMap<String, JAROW>>) o2);
                }

            } catch (ClassNotFoundException | IOException ex) {
            } finally {
                try {
                    fin1.close();
                    fin2.close();
                } catch (IOException ex) {
                }
                try {
                    ois1.close();
                    ois2.close();
                } catch (IOException ex) {
                }
            }
        } else {
            return false;
        }
        return true;
    }

    @Override
    public void writeInitClassifiers(int dataSize, HashMap<String, JAROW> trainedAttrClassifiers_0,
            HashMap<String, HashMap<String, JAROW>> trainedWordClassifiers_0) {
        String file1 = "cache/attrInitClassifiers" + "_" + dataSize;
        String file2 = "cache/wordInitClassifiers" + "_" + dataSize;
        FileOutputStream fout1 = null;
        ObjectOutputStream oos1 = null;
        FileOutputStream fout2 = null;
        ObjectOutputStream oos2 = null;
        try {
            System.out.print("Write initial classifiers...");
            fout1 = new FileOutputStream(file1);
            oos1 = new ObjectOutputStream(fout1);
            oos1.writeObject(trainedAttrClassifiers_0);

            fout2 = new FileOutputStream(file2);
            oos2 = new ObjectOutputStream(fout2);
            oos2.writeObject(trainedWordClassifiers_0);

        } catch (IOException ex) {
        } finally {
            try {
                fout1.close();
                fout2.close();
            } catch (IOException ex) {
            }
            try {
                oos1.close();
                oos2.close();
            } catch (IOException ex) {
            }
        }
    }

}

class InferE2EVectorsThread extends Thread {

    DatasetInstance di;
    E2E e2e;
    ConcurrentHashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>> predicateContentTrainingData;
    ConcurrentHashMap<DatasetInstance, HashMap<String, HashMap<String, ArrayList<Instance>>>> predicateWordTrainingData;

    InferE2EVectorsThread(DatasetInstance di, E2E e2e,
            ConcurrentHashMap<DatasetInstance, HashMap<String, ArrayList<Instance>>> predicateContentTrainingData,
            ConcurrentHashMap<DatasetInstance, HashMap<String, HashMap<String, ArrayList<Instance>>>> predicateWordTrainingData) {
        this.di = di;
        this.e2e = e2e;
        this.predicateContentTrainingData = predicateContentTrainingData;
        this.predicateWordTrainingData = predicateWordTrainingData;
    }

    public void run() {
        String predicate = di.getMeaningRepresentation().getPredicate();
        ArrayList<Action> refSequence = di.getDirectReferenceSequence();
        HashSet<String> attrValuesAlreadyMentioned = new HashSet<>();
        HashSet<String> attrValuesToBeMentioned = new HashSet<>();
        for (String attribute : di.getMeaningRepresentation().getAttributeValues().keySet()) {
            for (String value : di.getMeaningRepresentation().getAttributeValues().get(attribute)) {
                attrValuesToBeMentioned.add(attribute.toLowerCase() + "=" + value.toLowerCase());
            }
        }
        ArrayList<String> attributeSequence = new ArrayList<>();
        String attrValue = "";
        for (int w = 0; w < refSequence.size(); w++) {
            if (!refSequence.get(w).getAttribute().equals(Action.TOKEN_PUNCT)
                    && !refSequence.get(w).getAttribute().equals(attrValue)) {
                if (!attrValue.isEmpty()) {
                    attrValuesToBeMentioned.remove(attrValue);
                }
                // Create the feature and cost vector
                Instance contentTrainingVector = e2e.createContentInstance(predicate, refSequence.get(w).getAttribute(), attributeSequence,
                        attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), e2e.getAvailableContentActions());
                if (contentTrainingVector != null) {
                    predicateContentTrainingData.get(di).get(predicate).add(contentTrainingVector);
                }
                attributeSequence.add(refSequence.get(w).getAttribute());

                attrValue = refSequence.get(w).getAttribute();
                if (!attrValue.isEmpty()) {
                    attrValuesAlreadyMentioned.add(attrValue);
                    attrValuesToBeMentioned.remove(attrValue);
                }
            }
        }
        //reset track 
        attrValuesAlreadyMentioned = new HashSet<>();
        attrValuesToBeMentioned = new HashSet<>();
        for (String attribute : di.getMeaningRepresentation().getAttributeValues().keySet()) {
            for (String value : di.getMeaningRepresentation().getAttributeValues().get(attribute)) {
                attrValuesToBeMentioned.add(attribute.toLowerCase() + "=" + value.toLowerCase());
            }
        }
        ArrayList<String> attrs = new ArrayList<>();
        boolean isValueMentioned = false;
        // The value that we currently need to mention
        String valueTBM = "";
        // These track the content (attribute/value pairs)
        attrValue = "";
        // Time-step counter
        int a = -1;
        // This tracks the subphrase consisting of the words generated for the current content action
        ArrayList<String> subPhrase = new ArrayList<>();
        // For every step of the sequence
        for (int w = 0; w < refSequence.size(); w++) {
            if (!refSequence.get(w).getAttribute().equals(Action.TOKEN_PUNCT)) {
                // If this action does not belong to the current content, we need to update the trackers and switch to the new content action
                if (!refSequence.get(w).getAttribute().equals(attrValue)) {
                    a++;
                    if (!attrValue.isEmpty()) {
                        attrValuesToBeMentioned.remove(attrValue);
                    }
                    attrs.add(refSequence.get(w).getAttribute());

                    attrValue = refSequence.get(w).getAttribute();
                    subPhrase = new ArrayList<>();
                    isValueMentioned = false;
                    valueTBM = "";
                    if (attrValue.contains("=")) {
                        valueTBM = attrValue.substring(attrValue.indexOf('=') + 1);
                    }
                    if (valueTBM.isEmpty()) {
                        isValueMentioned = true;
                    }
                }

                // If it's not the end of the ActionSequence
                if (!attrValue.equals(Action.TOKEN_END)) {
                    // The subsequence of content actions we have generated for so far
                    ArrayList<String> predictedAttributesForInstance = new ArrayList<>();
                    for (int i = 0; i < attrs.size() - 1; i++) {
                        predictedAttributesForInstance.add(attrs.get(i));
                    }
                    // ...exclusive of the current content action

                    if (!attrs.get(attrs.size() - 1).equals(attrValue)) {
                        predictedAttributesForInstance.add(attrs.get(attrs.size() - 1));
                    }
                    // The subsequence of content actions we will generated for after the current content action
                    ArrayList<String> nextAttributesForInstance = new ArrayList<>(attributeSequence.subList(a + 1, attributeSequence.size()));
                    // Create the feature and cost vector
                    Instance wordTrainingVector = e2e.createWordInstance(predicate, refSequence.get(w), predictedAttributesForInstance,
                            new ArrayList<>(refSequence.subList(0, w)), nextAttributesForInstance, attrValuesAlreadyMentioned,
                            attrValuesToBeMentioned, isValueMentioned, e2e.getAvailableWordActions().get(predicate), di.getMeaningRepresentation());

                    if (wordTrainingVector != null) {
                        String attribute = attrValue;
                        if (attribute.contains("=")) {
                            attribute = attrValue.substring(0, attrValue.indexOf('='));
                        }
                        if (!predicateWordTrainingData.get(di).containsKey(predicate)) {
                            predicateWordTrainingData.get(di).put(predicate, new HashMap<String, ArrayList<Instance>>());
                        }
                        if (!predicateWordTrainingData.get(di).get(predicate).containsKey(attribute)) {
                            predicateWordTrainingData.get(di).get(predicate).put(attribute, new ArrayList<Instance>());
                        }
                        predicateWordTrainingData.get(di).get(predicate).get(attribute).add(wordTrainingVector);
                        if (!refSequence.get(w).getWord().equals(Action.TOKEN_START)
                                && !refSequence.get(w).getWord().equals(Action.TOKEN_END)) {
                            subPhrase.add(refSequence.get(w).getWord());
                        }
                    }

                    // Check if we have mentioned the value of the current content action
                    if (!isValueMentioned) {
                        // If the value is a variable, we just check if the word action we just generated is that variable
                        if (refSequence.get(w).getWord().startsWith(Action.TOKEN_X)
                                && (valueTBM.matches("[xX][0-9]+") || valueTBM.matches("\"[xX][0-9]+\"")
                                || valueTBM.startsWith(Action.TOKEN_X))) {
                            isValueMentioned = true;
                            // Otherwise
                        } else if (!refSequence.get(w).getWord().startsWith(Action.TOKEN_X)
                                && !(valueTBM.matches("[xX][0-9]+") || valueTBM.matches("\"[xX][0-9]+\"")
                                || valueTBM.startsWith(Action.TOKEN_X))) {
                            // We form the key for the value, as it appears in the valueAlignments collection
                            String valueToCheck = valueTBM;

                            if (valueToCheck.equals("no")
                                    || valueToCheck.equals("yes")
                                    || valueToCheck.equals("yes or no")
                                    || valueToCheck.equals("none")
                                    || valueToCheck.equals("empty")) {
                                String attribute = attrValue;
                                if (attribute.contains("=")) {
                                    attribute = attrValue.substring(0, attrValue.indexOf('='));
                                }
                                valueToCheck = attribute + ":" + valueTBM;
                            }
                            // We look up the value in all the value alignments we have made during the parsing of the data, and see if it is mentioned in the subphrase
                            // Note that the value may be formed by multiple word actions
                            if (!valueToCheck.equals("empty:empty")
                                    && e2e.getValueAlignments().containsKey(valueToCheck)) {
                                for (ArrayList<String> alignedStr : e2e.getValueAlignments().get(valueToCheck).keySet()) {
                                    if (e2e.endsWith(subPhrase, alignedStr)) {
                                        isValueMentioned = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if (isValueMentioned) {
                            attrValuesAlreadyMentioned.add(attrValue);
                            attrValuesToBeMentioned.remove(attrValue);
                        }
                    }
                    // We also check if we have inadvertedly mentioned some other pending value (not the current one)
                    String mentionedAttrValue = "";
                    if (!refSequence.get(w).getWord().startsWith(Action.TOKEN_X)) {
                        for (String attrValueTBM : attrValuesToBeMentioned) {
                            if (attrValueTBM.contains("=")) {
                                String value = attrValueTBM.substring(attrValueTBM.indexOf('=') + 1);

                                if (!value.startsWith(Action.TOKEN_X)) {
                                    String valueToCheck = value;
                                    if (valueToCheck.equals("no")
                                            || valueToCheck.equals("yes")
                                            || valueToCheck.equals("yes or no")
                                            || valueToCheck.equals("none")
                                            || valueToCheck.equals("empty")) {
                                        valueToCheck = attrValueTBM.replace("=", ":");
                                    }
                                    if (!valueToCheck.equals("empty:empty")
                                            && e2e.getValueAlignments().containsKey(valueToCheck)) {
                                        for (ArrayList<String> alignedStr : e2e.getValueAlignments().get(valueToCheck).keySet()) {
                                            if (e2e.endsWith(subPhrase, alignedStr)) {
                                                mentionedAttrValue = attrValueTBM;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (!mentionedAttrValue.isEmpty()) {
                        attrValuesAlreadyMentioned.add(mentionedAttrValue);
                        attrValuesToBeMentioned.remove(mentionedAttrValue);
                    }
                }
            }
        }

    }
}

class EvaluatorThread extends Thread {

    DatasetInstance di;
    E2E e2e;

    HashMap<String, JAROW> classifierAttrs;
    HashMap<String, HashMap<String, JAROW>> classifierWords;
                    
    ConcurrentHashMap<DatasetInstance, ArrayList<ScoredFeaturizedTranslation<IString, String>>> generations;
    ConcurrentHashMap<DatasetInstance, ArrayList<Action>> generationActions;
    ConcurrentHashMap<DatasetInstance, ArrayList<ArrayList<Sequence<IString>>>> finalReferences;
    ConcurrentHashMap<DatasetInstance, ArrayList<String>> finalReferencesWordSequences;
    ConcurrentHashMap<DatasetInstance, String> predictedWordSequences_overAllPredicates;
    ConcurrentHashMap<DatasetInstance, ArrayList<String>> allPredictedWordSequences;
    ConcurrentHashMap<DatasetInstance, ArrayList<String>> allPredictedMRStr;
    ConcurrentHashMap<DatasetInstance, ArrayList<ArrayList<String>>> allPredictedReferences;
    ConcurrentHashMap<DatasetInstance, HashMap<String, Double>> attrCoverage;

    ConcurrentHashMap<DatasetInstance, HashMap<String, HashSet<String>>> abstractMRsToMRs;
    
    EvaluatorThread(DatasetInstance di, E2E e2e,            
        HashMap<String, JAROW> classifierAttrs,
        HashMap<String, HashMap<String, JAROW>> classifierWords,
        ConcurrentHashMap<DatasetInstance, ArrayList<ScoredFeaturizedTranslation<IString, String>>> generations,
        ConcurrentHashMap<DatasetInstance, ArrayList<Action>> generationActions,
        ConcurrentHashMap<DatasetInstance, ArrayList<ArrayList<Sequence<IString>>>> finalReferences,
        ConcurrentHashMap<DatasetInstance, ArrayList<String>> finalReferencesWordSequences,
        ConcurrentHashMap<DatasetInstance, String> predictedWordSequences_overAllPredicates,
        ConcurrentHashMap<DatasetInstance, ArrayList<String>> allPredictedWordSequences,
        ConcurrentHashMap<DatasetInstance, ArrayList<String>> allPredictedMRStr,
        ConcurrentHashMap<DatasetInstance, ArrayList<ArrayList<String>>> allPredictedReferences,
        ConcurrentHashMap<DatasetInstance, HashMap<String, Double>> attrCoverage,
        ConcurrentHashMap<DatasetInstance, HashMap<String, HashSet<String>>> abstractMRsToMRs) {
        
        this.di = di;
        this.e2e = e2e;
        this.classifierAttrs = classifierAttrs;
        this.classifierWords = classifierWords;
        this.generations = generations;
        this.generationActions = generationActions;
        this.finalReferences = finalReferences;
        this.finalReferencesWordSequences = finalReferencesWordSequences;
        this.predictedWordSequences_overAllPredicates = predictedWordSequences_overAllPredicates;
        this.allPredictedWordSequences = allPredictedWordSequences;
        this.allPredictedMRStr = allPredictedMRStr;
        this.allPredictedReferences = allPredictedReferences;
        this.attrCoverage = attrCoverage;
        this.abstractMRsToMRs = abstractMRsToMRs;
        
        this.generations.put(di, new ArrayList<ScoredFeaturizedTranslation<IString, String>>());
        this.generationActions.put(di, new ArrayList<Action>());
        this.finalReferences.put(di, new ArrayList<ArrayList<Sequence<IString>>>());
        this.finalReferencesWordSequences.put(di, new ArrayList<String>());
        this.allPredictedWordSequences.put(di, new ArrayList<String>());
        this.allPredictedMRStr.put(di, new ArrayList<String>());
        this.allPredictedReferences.put(di, new ArrayList<ArrayList<String>>());
        this.attrCoverage.put(di, new HashMap<String, Double>());
        this.abstractMRsToMRs.put(di, new HashMap<String, HashSet<String>>());
    }

    public void run() {
        String predicate = di.getMeaningRepresentation().getPredicate();
        ArrayList<Action> predictedActionList = new ArrayList<>();
        ArrayList<Action> predictedWordList = new ArrayList<>();

        //PHRASE GENERATION EVALUATION
        String predictedAttr = "";
        ArrayList<String> predictedAttrValues = new ArrayList<>();

        HashSet<String> attrValuesToBeMentioned = new HashSet<>();
        HashSet<String> attrValuesAlreadyMentioned = new HashSet<>();
        for (String attribute : di.getMeaningRepresentation().getAttributeValues().keySet()) {
            for (String value : di.getMeaningRepresentation().getAttributeValues().get(attribute)) {
                attrValuesToBeMentioned.add(attribute.toLowerCase() + "=" + value.toLowerCase());
            }
        }
        if (attrValuesToBeMentioned.isEmpty()) {
            attrValuesToBeMentioned.add("empty=empty");
        }
        // generate attribute feature vectors till the end 
        while (!predictedAttr.equals(Action.TOKEN_END) && predictedAttrValues.size() < e2e.getMaxContentSequenceLength()) {
            if (!predictedAttr.isEmpty()) {
                attrValuesToBeMentioned.remove(predictedAttr);
            }
            if (!attrValuesToBeMentioned.isEmpty()) {
                Instance attrTrainingVector = e2e.createContentInstance(predicate, "@TOK@", predictedAttrValues, attrValuesAlreadyMentioned, attrValuesToBeMentioned, di.getMeaningRepresentation(), e2e.getAvailableContentActions());

                if (attrTrainingVector != null) {
                    Prediction predictAttr = classifierAttrs.get(predicate).predict(attrTrainingVector);
                    if (predictAttr.getLabel() != null) {
                        predictedAttr = predictAttr.getLabel().trim();

                        if (!classifierAttrs.get(predicate).getCurrentWeightVectors().keySet().containsAll(di.getMeaningRepresentation().getAttributeValues().keySet())) {
                            System.out.println("MR ATTR NOT IN CLASSIFIERS");
                            System.out.println(classifierAttrs.get(predicate).getCurrentWeightVectors().keySet());
                        }
                        String predictedValue = "";
                        if (!predictedAttr.equals(Action.TOKEN_END)) {
                            predictedValue = e2e.chooseNextValue(predictedAttr, attrValuesToBeMentioned);

                            HashSet<String> rejectedAttrs = new HashSet<>();
                            while (predictedValue.isEmpty() && (!predictedAttr.equals(Action.TOKEN_END) || (predictedAttrValues.isEmpty() && classifierAttrs.get(predicate).getCurrentWeightVectors().keySet().containsAll(di.getMeaningRepresentation().getAttributeValues().keySet())))) {
                                rejectedAttrs.add(predictedAttr);

                                predictedAttr = Action.TOKEN_END;
                                double maxScore = -Double.MAX_VALUE;
                                for (String attr : predictAttr.getLabel2Score().keySet()) {
                                    if (!rejectedAttrs.contains(attr)
                                            && (Double.compare(predictAttr.getLabel2Score().get(attr), maxScore) > 0)) {
                                        maxScore = predictAttr.getLabel2Score().get(attr);
                                        predictedAttr = attr;
                                    }
                                }
                                if (!predictedAttr.equals(Action.TOKEN_END)) {
                                    predictedValue = e2e.chooseNextValue(predictedAttr, attrValuesToBeMentioned);
                                }
                            }
                        }
                        if (!predictedAttr.equals(Action.TOKEN_END)) {
                            predictedAttr += "=" + predictedValue;
                        }
                        predictedAttrValues.add(predictedAttr);
                        if (!predictedAttr.isEmpty()) {
                            attrValuesAlreadyMentioned.add(predictedAttr);
                            attrValuesToBeMentioned.remove(predictedAttr);
                        }
                    } else {
                        predictedAttr = Action.TOKEN_END;
                        predictedAttrValues.add(predictedAttr);
                    }
                } else {
                    predictedAttr = Action.TOKEN_END;
                    predictedAttrValues.add(predictedAttr);
                }
            } else {
                predictedAttr = Action.TOKEN_END;
                predictedAttrValues.add(predictedAttr);
            }
        }

        //WORD SEQUENCE EVALUATION
        predictedAttr = "";
        ArrayList<String> predictedAttributes = new ArrayList<>();

        attrValuesToBeMentioned = new HashSet<>();
        attrValuesAlreadyMentioned = new HashSet<>();
        HashMap<String, ArrayList<String>> valuesToBeMentioned = new HashMap<>();
        for (String attribute : di.getMeaningRepresentation().getAttributeValues().keySet()) {
            for (String value : di.getMeaningRepresentation().getAttributeValues().get(attribute)) {
                attrValuesToBeMentioned.add(attribute.toLowerCase() + "=" + value.toLowerCase());
            }
            valuesToBeMentioned.put(attribute, new ArrayList<>(di.getMeaningRepresentation().getAttributeValues().get(attribute)));
        }
        if (attrValuesToBeMentioned.isEmpty()) {
            attrValuesToBeMentioned.add("empty=empty");
        }
        HashSet<String> attrValuesToBeMentionedCopy = new HashSet<>(attrValuesToBeMentioned);

        int a = -1;
        for (String attrValue : predictedAttrValues) {
            a++;
            if (!attrValue.equals(Action.TOKEN_END)) {
                String attribute = attrValue.split("=")[0];
                predictedAttributes.add(attrValue);

                //GENERATE PHRASES
                if (!attribute.equals(Action.TOKEN_END)) {
                    if (classifierWords.get(predicate).containsKey(attribute)) {
                        ArrayList<String> nextAttributesForInstance = new ArrayList<>(predictedAttrValues.subList(a + 1, predictedAttrValues.size()));
                        String predictedWord = "";

                        boolean isValueMentioned = false;
                        String valueTBM = "";
                        if (attrValue.contains("=")) {
                            valueTBM = attrValue.substring(attrValue.indexOf('=') + 1);
                        }
                        if (valueTBM.isEmpty()) {
                            isValueMentioned = true;
                        }
                        ArrayList<String> subPhrase = new ArrayList<>();
                        while (!predictedWord.equals(Action.TOKEN_END) && predictedWordList.size() < e2e.getMaxWordSequenceLength()) {
                            ArrayList<String> predictedAttributesForInstance = new ArrayList<>();
                            for (int i = 0; i < predictedAttributes.size() - 1; i++) {
                                predictedAttributesForInstance.add(predictedAttributes.get(i));
                            }
                            if (!predictedAttributes.get(predictedAttributes.size() - 1).equals(attrValue)) {
                                predictedAttributesForInstance.add(predictedAttributes.get(predictedAttributes.size() - 1));
                            }
                            Instance wordTrainingVector = e2e.createWordInstance(predicate, new Action("@TOK@", attrValue), predictedAttributesForInstance, predictedActionList, nextAttributesForInstance, attrValuesAlreadyMentioned, attrValuesToBeMentioned, isValueMentioned, e2e.getAvailableWordActions().get(predicate), di.getMeaningRepresentation());

                            if (wordTrainingVector != null
                                    && classifierWords.get(predicate) != null) {
                                if (classifierWords.get(predicate).get(attribute) != null) {
                                    Prediction predictWord = classifierWords.get(predicate).get(attribute).predict(wordTrainingVector);
                                    if (predictWord.getLabel() != null) {
                                        predictedWord = predictWord.getLabel().trim();
                                        while (predictedWord.equals(Action.TOKEN_END) && !predictedActionList.isEmpty() && predictedActionList.get(predictedActionList.size() - 1).getWord().equals(Action.TOKEN_END)) {
                                            double maxScore = -Double.MAX_VALUE;
                                            for (String word : predictWord.getLabel2Score().keySet()) {
                                                if (!word.equals(Action.TOKEN_END)
                                                        && (Double.compare(predictWord.getLabel2Score().get(word), maxScore) > 0)) {
                                                    maxScore = predictWord.getLabel2Score().get(word);
                                                    predictedWord = word;
                                                }
                                            }
                                        }

                                        predictedActionList.add(new Action(predictedWord, attrValue));

                                        if (!predictedWord.equals(Action.TOKEN_START)
                                                && !predictedWord.equals(Action.TOKEN_END)) {
                                            subPhrase.add(predictedWord);
                                            predictedWordList.add(new Action(predictedWord, attrValue));
                                        }
                                    } else {
                                        predictedWord = Action.TOKEN_END;
                                        predictedActionList.add(new Action(predictedWord, attrValue));
                                    }
                                } else {
                                    predictedWord = Action.TOKEN_END;
                                    predictedActionList.add(new Action(predictedWord, attrValue));
                                }

                            }
                            if (!isValueMentioned) {
                                if (!predictedWord.equals(Action.TOKEN_END)) {
                                    if (predictedWord.startsWith(Action.TOKEN_X)
                                            && (valueTBM.matches("\"[xX][0-9]+\"")
                                            || valueTBM.matches("[xX][0-9]+")
                                            || valueTBM.startsWith(Action.TOKEN_X))) {
                                        isValueMentioned = true;
                                    } else if (!predictedWord.startsWith(Action.TOKEN_X)
                                            && !(valueTBM.matches("\"[xX][0-9]+\"")
                                            || valueTBM.matches("[xX][0-9]+")
                                            || valueTBM.startsWith(Action.TOKEN_X))) {
                                        String valueToCheck = valueTBM;
                                        if (valueToCheck.equals("no")
                                                || valueToCheck.equals("yes")
                                                || valueToCheck.equals("yes or no")
                                                || valueToCheck.equals("none")
                                                //|| valueToCheck.equals("dont_care")
                                                || valueToCheck.equals("empty")) {
                                            if (attribute.contains("=")) {
                                                valueToCheck = attribute.replace("=", ":");
                                            } else {
                                                valueToCheck = attribute + ":" + valueTBM;
                                            }
                                        }
                                        if (!valueToCheck.equals("empty:empty")
                                                && e2e.getValueAlignments().containsKey(valueToCheck)) {
                                            for (ArrayList<String> alignedStr : e2e.getValueAlignments().get(valueToCheck).keySet()) {
                                                if (e2e.endsWith(subPhrase, alignedStr)) {
                                                    isValueMentioned = true;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                                if (isValueMentioned) {
                                    attrValuesAlreadyMentioned.add(attrValue);
                                    attrValuesToBeMentioned.remove(attrValue);
                                }
                            }
                            String mentionedAttrValue = "";
                            if (!predictedWord.startsWith(Action.TOKEN_X)) {
                                for (String attrValueTBM : attrValuesToBeMentioned) {
                                    if (attrValueTBM.contains("=")) {
                                        String value = attrValueTBM.substring(attrValueTBM.indexOf('=') + 1);
                                        if (!(value.matches("\"[xX][0-9]+\"")
                                                || value.matches("[xX][0-9]+")
                                                || value.startsWith(Action.TOKEN_X))) {
                                            String valueToCheck = value;
                                            if (valueToCheck.equals("no")
                                                    || valueToCheck.equals("yes")
                                                    || valueToCheck.equals("yes or no")
                                                    || valueToCheck.equals("none")
                                                    //|| valueToCheck.equals("dont_care")
                                                    || valueToCheck.equals("empty")) {
                                                valueToCheck = attrValueTBM.replace("=", ":");
                                            }
                                            if (!valueToCheck.equals("empty:empty")
                                                    && e2e.getValueAlignments().containsKey(valueToCheck)) {
                                                for (ArrayList<String> alignedStr : e2e.getValueAlignments().get(valueToCheck).keySet()) {
                                                    if (e2e.endsWith(subPhrase, alignedStr)) {
                                                        mentionedAttrValue = attrValueTBM;
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if (!mentionedAttrValue.isEmpty()) {
                                attrValuesAlreadyMentioned.add(mentionedAttrValue);
                                attrValuesToBeMentioned.remove(mentionedAttrValue);
                            }
                        }
                        if (predictedWordList.size() >= e2e.getMaxWordSequenceLength()
                                && !predictedActionList.get(predictedActionList.size() - 1).getWord().equals(Action.TOKEN_END)) {
                            predictedWord = Action.TOKEN_END;
                            predictedActionList.add(new Action(predictedWord, attrValue));
                        }
                    } else {
                        String predictedWord = Action.TOKEN_END;
                        predictedActionList.add(new Action(predictedWord, attrValue));
                    }
                }
            }
        }
        ArrayList<String> predictedAttrs = new ArrayList<>();
        predictedAttrValues.forEach((attributeValuePair) -> {
            predictedAttrs.add(attributeValuePair.split("=")[0]);
        });

        //System.out.println("");
        String predictedWordSequence = e2e.postProcessWordSequence(di.getMeaningRepresentation(), predictedActionList);

        //System.out.println(predictedWordSequence);
        //System.out.println(di.getDirectReference());
        //System.out.println("");
        ArrayList<String> predictedAttrList = e2e.getPredictedAttrList(predictedActionList);
        if (attrValuesToBeMentionedCopy.size() != 0.0) {
            double missingAttrs = 0.0;
            missingAttrs = attrValuesToBeMentionedCopy.stream().filter((attr) -> (!predictedAttrList.contains(attr))).map((_item) -> 1.0).reduce(missingAttrs, (accumulator, _item) -> accumulator + _item);
            double attrSize = attrValuesToBeMentionedCopy.size();
            attrCoverage.get(di).put(predictedWordSequence, missingAttrs / attrSize);
        }

        allPredictedWordSequences.get(di).add(predictedWordSequence);
        allPredictedMRStr.get(di).add(di.getMeaningRepresentation().getMRstr());
        predictedWordSequences_overAllPredicates.put(di, predictedWordSequence);

        if (!abstractMRsToMRs.containsKey(di.getMeaningRepresentation().getAbstractMR())) {
            abstractMRsToMRs.get(di).put(di.getMeaningRepresentation().getAbstractMR(), new HashSet<String>());
        }
        abstractMRsToMRs.get(di).get(di.getMeaningRepresentation().getAbstractMR()).add(di.getMeaningRepresentation().getMRstr());

        Sequence<IString> translation = IStrings.tokenize(NISTTokenizer.tokenize(predictedWordSequence.toLowerCase()));
        ScoredFeaturizedTranslation<IString, String> tran = new ScoredFeaturizedTranslation<>(translation, null, 0);
        generations.get(di).add(tran);
        generationActions.put(di, predictedActionList);

        ArrayList<Sequence<IString>> references = new ArrayList<>();
        ArrayList<String> referencesStrings = new ArrayList<>();

        if (e2e.getPerformEvaluationOn().equals("valid") || e2e.getPerformEvaluationOn().equals("train")) {
            for (String ref : di.getEvaluationReferences()) {
                referencesStrings.add(ref);
                references.add(IStrings.tokenize(NISTTokenizer.tokenize(ref)));
            }
        } else {
            //references = wenEvaluationReferenceSequences.get(di.getMeaningRepresentation().getMRstr());
            //referencesStrings = wenEvaluationReferences.get(di.getMeaningRepresentation().getMRstr());
            //if (references == null) {
            references = new ArrayList<>();
            referencesStrings = new ArrayList<>();
            for (String ref : di.getEvaluationReferences()) {
                referencesStrings.add(ref);
                references.add(IStrings.tokenize(NISTTokenizer.tokenize(ref)));
            }
            //}
        }
        allPredictedReferences.get(di).add(referencesStrings);
        finalReferencesWordSequences.put(di, referencesStrings);
        finalReferences.get(di).add(references);
    }
}
