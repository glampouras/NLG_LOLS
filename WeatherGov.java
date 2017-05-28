package WeatherGov ;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
//import java.util.regex.Matcher;
import java.util.regex.Pattern;

import gnu.trove.map.hash.TObjectDoubleHashMap;
import jarow.Instance;
import similarity_measures.Levenshtein;
import simpleLM.WeatherSimpleLM;










public class WeatherGov   {
	
	HashMap<String,HashMap<String,HashSet<String>>> attributeFieldValuePairs = new HashMap<>();
	Boolean useAlignmentData = true;
	HashSet<String> attributes = new HashSet<>();
	HashMap<String,HashSet<String>> attributeFields = new HashMap<>();
	HashMap<String,HashSet<String>> fieldValues = new HashMap<>();
	int maxWordSequenceLength = 0;
	int maxFieldSequenceLength = 0;
	int maxAttributeSequenceLength = 0;
	ArrayList<WeatherDatasetInstance> DatasetInstances = new ArrayList<>();
	ArrayList<WeatherDatasetInstance> testingData = new ArrayList<>();
	ArrayList<WeatherDatasetInstance> trainingData = new ArrayList<>();
	ArrayList<WeatherDatasetInstance> validationData = new ArrayList<>();
	
	HashMap<String,HashMap<ArrayList<String>,Double>> valueAlignments = new HashMap<>();
	ArrayList<ArrayList<String>> attributeFieldValueSequence = new ArrayList<>();
	HashMap<ArrayList<WeatherAction>,WeatherAction> punctuationPatterns = new HashMap<>();
	WeatherSimpleLM wordLMs;
	WeatherSimpleLM attrLMs;
	WeatherSimpleLM fieldLMs;
	HashSet<String> availableAttr ;
	HashMap<String,HashSet<String>> availableFieldPerAttr ;
	HashMap<String, HashMap<String,HashSet<WeatherAction>>> availableWordAction ;
	private ArrayList<Instance> attributeTrainingData;
	private HashMap<String,ArrayList<Instance>> fieldTrainingData;
	private HashMap<String,HashMap<String,ArrayList<Instance>>> wordTrainingData;
	
	//boolean debug = true;
	public static void main(String[] args){
		
		WeatherGov w = new WeatherGov();
		w.parseDataset();
		w.createTrainingData();
		
		
		
		
		
		
		
		
	}
	@SuppressWarnings("unused")
	public void parseDataset(){
		
		ArrayList<String> fileNames = readInDataName();
		
		if ((true&&!loadLists())) {
			createLists(fileNames);
			writeLists();
        }
		//System.out.println(DatasetInstances.get(200).state);
		String meiTestingData = "/Users/chenmingjie/Documents/workspace/JLOLS_NLG-master/weather-data/testing_data.txt";
		BufferedReader br =null ;
		try {
			br = new BufferedReader(new FileReader(meiTestingData));
		} catch (FileNotFoundException e) {
			//
			e.printStackTrace();
		}
		try {
			//boolean newfile;
			String state = "";
			String city = "";
			String date = "";
			String id = "";
			String line = br.readLine();
			while(line!=null){
				String s = line;
				if(s.startsWith("file id")){
					id = s.split(" ")[2];
					//newfile = true;
					state = "";
					city = "";
					date = "";
					
					
				}
				else if(s.startsWith("city")){
					city = s.split(":")[1];
				}
				else if(s.startsWith("eventfile")){
					date = s.split(":")[1];
				}
				else if(s.startsWith("state")){
					state = s.split(":")[1];
					//System.out.println("id is "+id+" city is "+city+" state is "+state+"date is "+ date);
					/*
					DatasetInstances.stream().forEach((di)->{
						if(!state.equals("")
								&&!city.equals("")
								&&!date.equals("")){
							
						}
					});*/
					for(WeatherDatasetInstance di : DatasetInstances){
						//System.out.println(di.state+" "+di.city+ " "+di.date);
						if(!state.equals("")
								&&!city.equals("")
								&&!date.equals("")){
							if(di.city.equals(city.trim())
									&&di.state.equals(state.trim())
									&&di.date.equals(date.trim())){
								//System.out.println("1");
								testingData.add(di);
							}
						}
					}
				}
				line = br.readLine();
			}
		} catch (IOException e) {
			// 
			e.printStackTrace();
		}
		//System.out.println(testingData.size());
		ArrayList<WeatherDatasetInstance> restData = new ArrayList<>();
		DatasetInstances.stream().forEach(di->{
			if(!testingData.contains(di)){
				restData.add(di);
			}
		});
		for (int i = 0; i < restData.size(); i++) {
            if (i < 1000) {
                validationData.add(restData.get(i));
            } else {
                trainingData.add(restData.get(i));
            }
        }
		
		//System.out.println(validationData.size()+" "+trainingData.size());
		System.out.println("Training data size: " + trainingData.size());
        System.out.println("Validation data size: " + validationData.size());
        System.out.println("Test data size: " + testingData.size());
	}
	/*
	 * read in the file names 
	 * */
	public ArrayList<String> readInDataName(){
		ArrayList<String> fileNames = new ArrayList<>();
		final File folder = new File("weather-data/data");
		File[] listOfFiles = folder.listFiles();
		ArrayList<String> folderNames = new ArrayList<>() ;
		String folderRegex = "weather-data/data/%s";
		String folderName = "";
		for(int i = 0;i< listOfFiles.length;i++){
			if(listOfFiles[i].isDirectory()){
				
				folderName = String.format(folderRegex,listOfFiles[i].getName() );
				folderNames.add(folderName);
				
			}
		}
		String fileName ;
		for(String name : folderNames){
			File file = new File(name);
			File[] fileList = file.listFiles();
			for(int i = 0; i< fileList.length;i++){
				fileName = String.format(name + "/%s", fileList[i].getName());
				File file1 = new File(fileName);
				File[] fileList1 = file1.listFiles();
				if(fileList1 == null){
				}else{
					for(int j = 0; j<fileList1.length;j++){
						fileNames.add(String.format(fileName+"/%s", fileList1[j].getName()));
						
					}
				}
			}
		}		
		return(fileNames);
	}
	
	
	
	
	/*
	 * read in data and create MeaningRepresentation and DatasetInstances for each instance
	 * */
	public void createLists(ArrayList<String> fileNames){
		HashMap<String,ArrayList<String>>  docToEvents = new HashMap<>();
		HashMap<String,ArrayList<String>>  docToAlign = new HashMap<>();
		HashMap<String,ArrayList<String>>  docToText = new HashMap<>();
		BufferedReader br = null;
		for(String name : fileNames){
			if(name.contains(".events")){
				try {
					
					br = new BufferedReader(new FileReader(name));
				} catch (Exception e1) {
					// 
					e1.printStackTrace();
				
				}
				try {
					String line = br.readLine();
					//String line = er.readString();
					while (line != null) {
						String s = line;
						//String eventName = name.split(".")[0];
						if(!docToEvents.containsKey(name)){
							docToEvents.put(name, new ArrayList<>());
							docToEvents.get(name).add(s);
							
						}else{
							docToEvents.get(name).add(s);
						}		        
						line = br.readLine();
					}			
					

				} catch (Exception e) {
					// 
					e.printStackTrace();
					System.out.println("can not find file");
				}finally{
					try {
						br.close();
					} catch (IOException e) {
						// 
						e.printStackTrace();
					}
				}
			}else if(name.contains(".align")){
				try {
					br = new BufferedReader(new FileReader(name));
					//er = new EasyReader(name);
				} catch (Exception e1) {
					// 
					e1.printStackTrace();
				
				}
				try {
					String line = br.readLine();
					//String line = er.readString();
					while (line != null) {
						String s = line;
						//String alignName = name.split(".")[0];
						if(!docToAlign.containsKey(name)){
							docToAlign.put(name, new ArrayList<>());
							docToAlign.get(name).add(s);
							
						}else{
							docToAlign.get(name).add(s);
						}		        
						line = br.readLine();
					}			
					

				} catch (Exception e) {
					// 
					e.printStackTrace();
					System.out.println("can not find file");
				}finally{
					try {
						br.close();
					} catch (IOException e) {
						
						e.printStackTrace();
					}
				}
			}else if(name.contains(".text")){
				try {
					br = new BufferedReader(new FileReader(name));
					//er = new EasyReader(name);
				} catch (Exception e1) {
					
					e1.printStackTrace();
				
				}
				try {
					String line = br.readLine();
					//String line = er.readString();
					while (line != null) {
						String s = line;
						//String textName = name.split(".")[0];
						if(!docToText.containsKey(name)){
							docToText.put(name, new ArrayList<>());
							docToText.get(name).add(s);
							
						}else{
							docToText.get(name).add(s);
						}		        
						line = br.readLine();
					}			
					

				} catch (Exception e) {
					
					e.printStackTrace();
					System.out.println("can not find file");
				}finally{
					try {
						br.close();
					} catch (IOException e) {
						
						e.printStackTrace();
					}
				}
			}

			
		}
		if(docToEvents.isEmpty()||docToAlign.isEmpty()||docToText.isEmpty()){
			System.out.println("failed read in data");
			System.exit(0);
		}else{
			for(String key : docToEvents.keySet()){
				if(docToEvents.get(key).isEmpty()){
					System.out.println(key+":"+docToEvents.get(key));
				}
			}
			for(String key : docToAlign.keySet()){
				if(docToAlign.get(key).isEmpty()){
					System.out.println(key+":"+docToAlign.get(key));
				}
			}
			for(String key : docToText.keySet()){
				if(docToText.get(key).isEmpty()){
					System.out.println(key+":"+docToText.get(key));
				}
			}

		}
		//int numberFile=0;
		/* start for each instance , generate Meaning Representation and reference (dataInstance)
		 * 
		 * */
		int numberFile=0;
		for(String docName : docToEvents.keySet()){
			//if(numberFile%100==0){
			
			
			System.out.println("this is "+numberFile);
			String state = "";
			String city = "";
			String date = "";
			String[] components = docName.split("/");
			state = components[2];
			city = components[3];
			date = components[4];
			date = date.substring(0, date.indexOf("."));
			
			
			
			String attribute;
			ArrayList<String> ref = new ArrayList<>() ;
			String MRstr = "";
			ArrayList<String> align = new ArrayList<>();
			HashMap<String, HashMap<String,String>> attrFieldValue  = new HashMap<>();//for each instance 
			
			String textName = docName.substring(0, docName.indexOf("."))+".text";
			if(docToText.containsKey(textName)){
				for(String textLine : docToText.get(textName)){
					if(textLine!=null){
						ref.add(textLine+" ");
					}
				}
			}else{
				System.out.println("can not find text file"+ textName);
			}
			if(useAlignmentData){
				String alignName = docName.substring(0, docName.indexOf("."))+".align";
				if(docToAlign.containsKey(alignName)){
					for(String alignLine : docToAlign.get(alignName)){
						if(alignLine!=null){
							align.add(alignLine);
							//System.out.println(alignLine);
						}
					}
				}else{
					System.out.println("can not find align file"+alignName);
				}
				
			}else{
				/*if not use alignmentData ,something will be add here
				 * */
			}
			
			/*
			 * generate Meaning Representation 
			 * */
			
			for(String eventLine : docToEvents.get(docName)){
				String[] fields ;
				fields = eventLine.split("\\s+");
				
				if(!fields[fields.length-1].equals("@mode:--")){//filter out mode= -- , no use
					attribute = fields[0].split(":")[1];
					if(fields[1].split(":")[0].equals(".type")){
						attribute = attribute+"#"+fields[1].split(":")[1];
					}
					 
					attributes.add(attribute);
					if(!attributeFields.containsKey(attribute)){
						attributeFields.put(attribute, new HashSet<String>());
					}
					for(int i = 1;  i<fields.length;i++){
						/*
						 * remove the data: label: and Night , no use. 
						 * */					
						if(fields[i].contains(".date:")){
						}
						//System.out.println(fields[i]);
						else if(fields[i].contains(".label:")){
						}
						else if(fields[i].equals("Night")){
						}
						else{
							try{
								String value = "";
								String field = "";
								MRstr = MRstr + fields[i]+"  " ;
								field = fields[i].substring(1, fields[i].length()).split(":")[0];
								//System.out.println(field);
								/*
								 * some event file lost mode value , give them an "EMPTY" value
								 * */
								if(fields[i].substring(1, fields[i].length()).split(":").length==1){
									value = "empty";
								}else{
							
									value = fields[i].substring(1, fields[i].length()).split(":")[1];
									
								}
								
								/*
								 * change the max , mean , min , numbers to categories
								 * */
								if(field.equals("max")||field.equals("mean")||field.equals("min")){
									//System.out.println(field + ":"+ value);
								}
								if(field!=null&&value!=null){
									//populate attributeFieldValuePairs
									if(!attributeFieldValuePairs.containsKey(attribute)){
										attributeFieldValuePairs.put(attribute, new HashMap<>());
									} 
									if(!attributeFieldValuePairs.get(attribute).containsKey(field)){
										attributeFieldValuePairs.get(attribute).put(field, new HashSet<>());
									}
										
									attributeFieldValuePairs.get(attribute).get(field).add(value);
									
									//populate attrFieldValue
									//for every MR
									if(!attrFieldValue.containsKey(attribute)){
										
										attrFieldValue.put(attribute, new HashMap<>());
									}
									if(!attrFieldValue.get(attribute).containsKey(field)){
										attrFieldValue.get(attribute).put(field,value);
									}
									attributeFields.get(attribute).add(field);
									
									if(!fieldValues.containsKey(field)){
										fieldValues.put(field, new HashSet<>());
									}
									
									fieldValues.get(field).add(value);
								}
								
								
							}catch(Exception e){
								
								System.out.println(e);
								System.out.println(fields[i]+"   "+docName);
								
							}
						}
					}
				}
				
			}//end reading each file
			/*
			HashMap<String,HashSet<String>> textAttrIdAlignment = new  HashMap<>();
			HashSet<String> alignedAttrRecord = new HashSet<>();
			if(!attrFieldValue.isEmpty()
					&&!align.isEmpty()
					&&!ref.isEmpty()){
				for(int i =0; i<align.size();i++){
					if(!textAttrIdAlignment.containsKey(ref.get(i))){
						textAttrIdAlignment.put(ref.get(i),new HashSet<>());
					}
					
					//System.out.println(align.get(i));
					String[] id = align.get(i).split("\\s+");
					if(id!=null){
						if(!textAttrIdAlignment.containsKey(id[0])){
							textAttrIdAlignment.put(id[0], new HashSet<>());
							
						}
						
						for(int j = 1;j<id.length;j++){
							textAttrIdAlignment.get(id[0]).add(id[j]);
							alignedAttrRecord.add(id[j]);//this file contains what attribute 
							
						}
					}else{System.out.println("id is null!");}
				
				}
			}
			*/
			
			/*
			 * for each align file, check if retain the unique time attributes is right
			 * */
			/*
			String time = "";
			for(String id :alignedAttrRecord){
				
				if(attrFieldValue.containsKey(id)){
					if(attrFieldValue.get(id).get("type").equals("temperature")){
						time = attrFieldValue.get(id).get("time");
						
					}
					if(!attrFieldValue.get(id).get("time").equals(time)){
						System.out.println(attrFieldValue.get(id));
						System.out.println(time);
						System.out.println(docName);
					}
				}
			}
			*/
			//for each figure in the text, find the closest figure in the MR, and change it with a label e.g. @attribute+field
			String refStr = String.join("", ref);
			String[] words = refStr.replaceAll("([.,])", WeatherAction.TOKEN_PUNCT).split("\\s+");
			ArrayList<String> observedAttrFieldValueSequence = new ArrayList<>();
            ArrayList<String> observedWordSequence = new ArrayList<>();
            HashMap<String,String> deleMap = new HashMap<>();
			for(String w:words){
				if(w.matches("([0-9]+)")){
					String minValue = "";
					String minAttr = "";
					String minField = "";
					int min = 100;
					for(String attr:attrFieldValue.keySet()){
						for(String field:attrFieldValue.get(attr).keySet()){
							String value = attrFieldValue.get(attr).get(field);
							if(value.matches("[0-9]+")){
								if(Math.abs(Integer.parseInt(value)-Integer.parseInt(w))<min){
									min = Math.abs(Integer.parseInt(value)-Integer.parseInt(w));
									minValue = value;
									minField = field;
									minAttr = attr;
									
								}
							}
						}
					}
				if(!minValue.isEmpty()&&min<=5&&!minValue.contains("X")){	
					
					attrFieldValue.get(minAttr).put(minField, "X:"+minAttr+":"+minField);
					deleMap.put(minValue, "X:"+minAttr+":"+minField);
					if(MRstr.contains(minValue)){
						MRstr = MRstr.replace(minValue, "X:"+minAttr+":"+minField);
					}
					if(refStr.contains(w)){
						refStr = refStr.replace(w, "X:"+minAttr+":"+minField);
					}
					
					
					observedWordSequence.add("X:"+minAttr+":"+minField);
					
				}
				}else{
					observedWordSequence.add(w.trim().toLowerCase());
				}
			}
			
			
			//  for each instance build MR
			WeatherMeaningRepresentation MR = new WeatherMeaningRepresentation(attrFieldValue,MRstr,deleMap);
			
			//start build DatasetInstance
			
            
            /*
            for(String w : words){
            	//if((!w.isEmpty())&&(observedWordSequence.isEmpty()||w.trim().equals(observedWordSequence.get(observedWordSequence.size()-1)if((!w.isEmpty())&&(observedWordSequence.isEmpty()||w.trim().equals(observedWordS))){
            		observedWordSequence.add(w.trim().toLowerCase());
            	//}
            }*/
            //System.out.println(observedWordSequence);
            if(observedWordSequence.size()>maxWordSequenceLength){
            	maxWordSequenceLength = observedWordSequence.size();
            }
            ArrayList<String> wordToAttrFieldValueAlignment = new ArrayList<>();
            observedWordSequence.forEach((word) -> {
                if (word.trim().matches(WeatherAction.TOKEN_PUNCT) ){
                    wordToAttrFieldValueAlignment.add(WeatherAction.TOKEN_PUNCT);
                }else if(word.contains("X")){
                	wordToAttrFieldValueAlignment.add(word.split(":")[1]+"="+word.split(":")[2]+"="+word);
                }
                else {
                    wordToAttrFieldValueAlignment.add("[]");
                }
            });
            if(wordToAttrFieldValueAlignment.size()!=observedWordSequence.size()){
            	System.out.println("length not equal");
            }
            ArrayList<WeatherAction> directReferenceSequence = new ArrayList<>();
            for (int r = 0; r < observedWordSequence.size(); r++) {
            	
            	directReferenceSequence.add(new WeatherAction(observedWordSequence.get(r),wordToAttrFieldValueAlignment.get(r),wordToAttrFieldValueAlignment.get(r)));
            	
            } 
            
            /*
             * build DI
             * */
            
            WeatherDatasetInstance DI = new WeatherDatasetInstance(MR,directReferenceSequence,postProcessRef(MR, directReferenceSequence));
            DI.city  = city;
            DI.date = date;
            DI.state = state;
            /*
             * populate evaluationReferences combine all reference for DIs that have same MR
             * */
            DatasetInstances.stream().filter((existingDI)->(existingDI.getMR().getAbstractMR().equals(DI.getMR().getAbstractMR())))
            .map((existingDI)->{
            	
            	existingDI.getEvaluationReferences().addAll(DI.getEvaluationReferences());
            	return existingDI;
            }).forEachOrdered((existingDI)->{
            	DI.getEvaluationReferences().addAll(existingDI.getEvaluationReferences());
            });
            DatasetInstances.add(DI);
            
            /*
             * build  valueAlignment
             * */
            HashMap<String,HashMap<String,Double>> observedValueAlignments = new HashMap<>();
            
            MR.getAttrFieldValue().keySet().forEach((attr)->{
            	ArrayList<String> fields ;
            	fields = new ArrayList<>(MR.getAttrFieldValue().get(attr).keySet());
            	Collections.sort(fields);
            	fields.stream().forEach((field)->{
            		boolean isDigit;
                    boolean isTime;
            		isDigit = false;
            		isTime = false;
            		String valueToCompare;
            		if(!Pattern.matches("([0-9]+)", MR.getAttrFieldValue().get(attr).get(field))&&!
            				(field.equals("time")||field.equals("max")||field.equals("mean")||field.equals("min")
            						||Pattern.matches("([0-9]+)-([0-9]+)",MR.getAttrFieldValue().get(attr).get(field) ))
            				&&!MR.getAttrFieldValue().get(attr).get(field).contains("X")){
            			
            		/*
            		if(Pattern.matches("([0-9]+)", valueToCompare)){
            			//System.out.println(valueToCompare+" is digit");
            			isDigit = true;
            		}
            		if(field.equals("time")&&Pattern.matches("([0-9]+)-([0-9]+)",valueToCompare )){
            			//System.out.println(valueToCompare+" is time");
            			isTime = true;
            		}*/
            		valueToCompare = MR.getAttrFieldValue().get(attr).get(field);
            		if(valueToCompare.equals("windDir")){
            			valueToCompare = "wind Dir";
            		}
            		if(valueToCompare.equals("windSpeed")){
            			valueToCompare = "wind Speed";
            		}
            		if(valueToCompare.equals("windChill")){
            			valueToCompare = "wind Chill";
            		}
            		if(valueToCompare.equals("freezingRainChance")){
            			valueToCompare = "freezing Rain Chance";
            		}
            		if(valueToCompare.equals("freezingRainChance")){
            			valueToCompare = "freezing Rain Chance";
            		}
            		if(valueToCompare.equals("precipPotential")){
            			valueToCompare = "precip Potential";
            		}
            		if(valueToCompare.equals("rainChance")){
            			valueToCompare = "rain Chance";
            		}
            		if(valueToCompare.equals("thunderChance")){
            			valueToCompare = "thunder Chance";
            		}
            		if(valueToCompare.equals("sleetChance")){
            			valueToCompare = "sleet Chance";
            		}
            		if(valueToCompare.equals("snowChance")){
            			valueToCompare = "snow Chance";
            		}
            		if(valueToCompare.equals("skyCover")){
            			valueToCompare = "sky Cover";
            		}
            		
            		observedValueAlignments.put(valueToCompare, new HashMap<>());
            		for(int n = 1;n<observedWordSequence.size();n++){
            			for(int r = 0;r<=observedWordSequence.size()-n;r++){
            				boolean compareAgainstNGram = true;
            				for(int j = 0;j<n;j++){
            					
            					if(observedWordSequence.get(j+r).equals(WeatherAction.TOKEN_PUNCT)
            								||observedWordSequence.get(r + j).equalsIgnoreCase("and")
            								||observedWordSequence.get(r + j).equalsIgnoreCase("or")
            								||observedWordSequence.get(r+j).matches("([0-9]+)")){
            						compareAgainstNGram = false;
            					}
            				}
            				if(!isDigit&&!isTime&&compareAgainstNGram){
            					String alignIndex = "";
            					String compare = "";
            					String backwardCompare = "";
            					for(int j = 0;j<n;j++){
            						alignIndex += (j+r)+" ";
            						compare += observedWordSequence.get((r+j));
            						backwardCompare = observedWordSequence.get(r + j) + backwardCompare;
            						 
            					}
            					alignIndex = alignIndex.trim();
            					Double distance = Levenshtein.getSimilarity(valueToCompare.toLowerCase(), compare.toLowerCase(), true);
            					Double backwardDistance = Levenshtein.getSimilarity(valueToCompare.toLowerCase(), backwardCompare.toLowerCase(), true);
            					if (backwardDistance > distance) {
                                    distance = backwardDistance;
                                }
            					//if (distance > 0.3) {
            						observedValueAlignments.get(valueToCompare).put(alignIndex, distance);
                                    
                                //}
            				}/*else if(isDigit&&!isTime&&(field.equals("max")||field.equals("min")||field.equals("mean"))){
            					
            					         				
            				}*/
            			}
            			
            		}/*
            		if(isDigit&&!isTime&&(field.equals("max")||field.equals("min")||field.equals("mean"))){
            			Pattern patt = Pattern.compile("([0-9]+)");
            			for(int k = 0;k<observedWordSequence.size();k++){
            				Matcher mat = patt.matcher(observedWordSequence.get(k));
            				if(mat.find()){
            					Integer figure = Integer.parseInt(mat.group());//figure in the text
            					Integer valueFigure = Integer.parseInt(valueToCompare);//figure in the MR value
            					if(Math.abs(figure-valueFigure)<=5){
            						observedValueAlignments.get(valueToCompare).put(Integer.toString(k),(-1)*(double)Math.abs(figure-valueFigure) );
            					}
            				}
            			}
            		}*/
            		}	
            	});
            		
            });
            //System.out.println(observedValueAlignments);
            HashSet<String> toRemove = new HashSet<>();
            for (String value : observedValueAlignments.keySet()) {
                if (observedValueAlignments.get(value).isEmpty()) {
                    toRemove.add(value);
                }
            }
            for (String value : toRemove) {
                observedValueAlignments.remove(value);
            }
            //split the numbers and word value
            while (!observedValueAlignments.keySet().isEmpty()) {
                // Find the best aligned nGram
                Double max = Double.NEGATIVE_INFINITY;
                String[] bestAlignment = new String[2];
                //numbers value 
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
                if(!valueAlignments.containsKey(bestAlignment[0])){
                	valueAlignments.put(bestAlignment[0], new HashMap<>());
                	
                }
                valueAlignments.get(bestAlignment[0]).put(alignedStr, max);
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
            attributeFieldValueSequence.add(observedAttrFieldValueSequence);
		//} //end if debug  
         numberFile++;
		}
		//end for each instance
	
	}
	public String postProcessRef(WeatherMeaningRepresentation mr, ArrayList<WeatherAction> directReferenceSequence){
		String cleanedWords = "";
		for(WeatherAction nlWord : directReferenceSequence){
			if(!nlWord.equals(new WeatherAction("","",WeatherAction.TOKEN_START))
					&&!nlWord.equals(new WeatherAction("","",WeatherAction.TOKEN_END))){
				if(nlWord.getWord().startsWith("X")){
					cleanedWords+=" "+mr.getDeleMap().get(nlWord.getWord());
				}else{
				cleanedWords += " " + nlWord.getWord();
				}
			}
			
			
			
		}
		if(!cleanedWords.endsWith(".")){
			cleanedWords = cleanedWords.trim()+ ".";
		}
		
		return cleanedWords;
	}
	public void writeLists(){
		
		String file1 = "cache/attributeFieldValuePairs" ;
        String file2 = "cache/attributes" ;
        String file3 = "cache/valueAlignments";
        String file4 = "cache/DatasetInstances";
        String file5 = "cache/maxWordSequenceLength";
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
        try {
            System.out.print("Write lists...");
            fout1 = new FileOutputStream(file1);
            oos1 = new ObjectOutputStream(fout1);
            oos1.writeObject(attributeFieldValuePairs);
            ///////////////////
            fout2 = new FileOutputStream(file2);
            oos2 = new ObjectOutputStream(fout2);
            oos2.writeObject(attributes);
            ///////////////////
            fout3 = new FileOutputStream(file3);
            oos3 = new ObjectOutputStream(fout3);
            oos3.writeObject(valueAlignments);
            ///////////////////
            fout4 = new FileOutputStream(file4);
            oos4 = new ObjectOutputStream(fout4);
            oos4.writeObject(DatasetInstances);
            ///////////////////
            fout5 = new FileOutputStream(file5);
            oos5 = new ObjectOutputStream(fout5);
            oos5.writeObject(maxWordSequenceLength);
            ///////////////////
            
          
        } catch (IOException ex) {
        } finally {
            try {
                fout1.close();
                fout2.close();
                fout3.close();
                fout4.close();
                fout5.close();
                
            } catch (IOException ex) {
            }
            try {
                oos1.close();
                oos2.close();
                oos3.close();
                oos4.close();
                oos5.close();
            } catch (IOException ex) {
            }
        }
	}
	@SuppressWarnings("unchecked")
	public boolean loadLists(){
		String file1 = "cache/attributeFieldValuePairs" ;
        String file2 = "cache/attributes" ;
        String file3 = "cache/valueAlignments";
        String file4 = "cache/DatasetInstances";
        String file5 = "cache/maxWordSequenceLength";
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
        if ((new File(file1)).exists()
                && (new File(file2)).exists()
                && (new File(file3)).exists()
                && (new File(file4)).exists()
                && (new File(file5)).exists()) {
        	try{
        		System.out.print("Load lists...");
        		fin1 = new FileInputStream(file1);
                ois1 = new ObjectInputStream(fin1);
                Object o1 = ois1.readObject();
                if (attributeFieldValuePairs.isEmpty()) {
                    if (o1 instanceof HashMap) {
                    	attributeFieldValuePairs = new HashMap<String,HashMap<String,HashSet<String>>>((Map<? extends String,? extends HashMap<String,HashSet<String>>>) o1);
                    }
                } else if (o1 instanceof ArrayList) {
                	attributeFieldValuePairs.putAll((Map<? extends String,? extends HashMap<String,HashSet<String>>>) o1);
                }
                fin2 = new FileInputStream(file2);
                ois2 = new ObjectInputStream(fin2);
                Object o2 = ois2.readObject();
                if(attributes.isEmpty()){
                	if(o2 instanceof HashSet){
                		attributes = new HashSet<String>((Collection<? extends String>)o2);
                	}
                }else{
                	attributes.addAll((Collection<? extends String>)o2);
                }
                fin3 = new FileInputStream(file3);
                ois3 = new ObjectInputStream(fin3);
                Object o3 = ois3.readObject();
                if(valueAlignments.isEmpty()){
                	if(valueAlignments instanceof HashMap){
                		valueAlignments = new HashMap<String, HashMap<ArrayList<String>,Double>>((Map<? extends String,? extends HashMap<ArrayList<String>, Double>>)o3);
                	}
                }else{
                	valueAlignments.putAll((Map<? extends String,? extends HashMap<ArrayList<String>, Double>>)o3);
                }
                fin4 = new FileInputStream(file4);
                ois4 = new ObjectInputStream(fin4);
                Object o4 = ois4.readObject();
                if(DatasetInstances instanceof ArrayList){
                	if(DatasetInstances.isEmpty()){
                		DatasetInstances = new ArrayList<WeatherDatasetInstance>((Collection<? extends WeatherDatasetInstance>)o4);
                	}
                }else{
                	DatasetInstances.addAll((Collection<? extends WeatherDatasetInstance>)o4);
                }
                fin5 = new FileInputStream(file5);
                ois5 = new ObjectInputStream(fin5);
                Object o5 = ois5.readObject();
                maxWordSequenceLength = (int)o5;
                System.out.println("done!");
                
        	}catch (ClassNotFoundException | IOException ex) {
        		
            }
        	return true;
        }else{
        	return false;
        }
	}
	public void createTrainingData(){
		createNaiveAlignments(trainingData);
		if(!loadLMs()){
			ArrayList<ArrayList<String>> LMWordTraining = new ArrayList<>();
			ArrayList<ArrayList<String>> LMFieldTraining = new ArrayList<>();
			ArrayList<ArrayList<String>> LMAttrTraining = new ArrayList<>();
			trainingData.stream().forEach((di)->{
				HashSet<ArrayList<WeatherAction>> seqs = new HashSet<>();
				seqs.add(di.getDirectReferenceSequence());
				seqs.forEach(seq->{
					ArrayList<String> wordSeq = new ArrayList<>();
					ArrayList<String> attrFieldSeq = new ArrayList<>();
					ArrayList<String> attrSeq = new ArrayList<>();
					wordSeq.add("@@");
					wordSeq.add("@@");
					attrSeq.add("@@");
					attrSeq.add("@@");
					attrFieldSeq.add("@@");
					attrFieldSeq.add("@@");
                
					for(int i=0;i<seq.size();i++){
						if(!seq.get(i).getAttribute().equals(WeatherAction.TOKEN_END)
							&&!seq.get(i).getField().equals(WeatherAction.TOKEN_END)
							&&!seq.get(i).getWord().equals(WeatherAction.TOKEN_END)
							&&!seq.get(i).getAttribute().equals(WeatherAction.TOKEN_PUNCT)
							&&!seq.get(i).getField().equals(WeatherAction.TOKEN_PUNCT)
							&&!seq.get(i).getWord().equals(WeatherAction.TOKEN_PUNCT)){
							wordSeq.add(seq.get(i).getWord());
						
						
						}
						if(!seq.get(i).getAttribute().equals(WeatherAction.TOKEN_PUNCT)){
							if (attrSeq.isEmpty()) {
								attrSeq.add(seq.get(i).getAttribute().split("=")[0]);
							} else if (!attrSeq.get(attrSeq.size() - 1).equals(seq.get(i).getAttribute().split("=")[0])) {
								attrSeq.add(seq.get(i).getAttribute().split("=")[0]);
							}
							if(attrFieldSeq.isEmpty()){
								attrFieldSeq.add(seq.get(i).getField().split("=")[0]
										+"="+seq.get(i).getField().split("=")[1]);
							}
						}
					
					}
					wordSeq.add(WeatherAction.TOKEN_END);
				
					LMWordTraining.add(wordSeq);
					LMAttrTraining.add(attrSeq);
					LMFieldTraining.add(attrFieldSeq);
					// we didn't use language model for field, may be will add it later
				});
			
			});
			wordLMs  = new WeatherSimpleLM(3);
			attrLMs = new WeatherSimpleLM(3);
			fieldLMs = new WeatherSimpleLM(3);
			wordLMs.trainOnStrings(LMWordTraining);
			attrLMs.trainOnStrings(LMAttrTraining);
			fieldLMs.trainOnStrings(LMFieldTraining);
			writeLMs();
		}
		// Go through the sequences in the data and populate the available content and word action dictionaries
        // We populate a distinct word dictionary for each attribute, 
		//and populate it with the words of word sequences whose corresponding content sequences contain that attribute
		availableAttr = new HashSet<>();
		availableFieldPerAttr = new HashMap<>();
		availableWordAction = new HashMap<>();
		for(WeatherDatasetInstance di : trainingData){
		
			
			for(String attribute : di.getMR().getAttrFieldValue().keySet()){
				if(!availableAttr.contains(attribute)){
					availableAttr.add(attribute);
					availableAttr.add(WeatherAction.TOKEN_END);
				}
				for(String field : di.getMR().getAttrFieldValue().get(attribute).keySet()){
					//String word = a.getWord();
					
					if(!availableFieldPerAttr.containsKey(attribute)){
						availableFieldPerAttr.put(attribute, new HashSet<>());
					}
					
					availableFieldPerAttr.put(WeatherAction.TOKEN_END, new HashSet<>());
					
					availableFieldPerAttr.get(attribute).add(attribute+"="+field);
					availableFieldPerAttr.get(attribute).add(WeatherAction.TOKEN_END);
					if(!availableWordAction.containsKey(attribute)){
						availableWordAction.put(attribute, new HashMap<>());
					}
					availableWordAction.put(WeatherAction.TOKEN_END, new HashMap<>());
					if(!availableWordAction.get(attribute).containsKey(attribute+"="+field)){
						availableWordAction.get(attribute).put(field, new HashSet<>());
					}
					availableWordAction.get(WeatherAction.TOKEN_END).put(WeatherAction.TOKEN_END, new HashSet<>());
					for(WeatherAction a : di.getDirectReferenceSequence()){
						if(!a.getWord().equals(WeatherAction.TOKEN_PUNCT)){
							availableWordAction.get(attribute).get(field).add(a);
							
						}
					}
					availableWordAction.get(attribute).get(field).add(new WeatherAction(attribute,field,WeatherAction.TOKEN_END));
					availableWordAction.get(WeatherAction.TOKEN_END).get(WeatherAction.TOKEN_END).add(new WeatherAction(WeatherAction.TOKEN_END
							,WeatherAction.TOKEN_END,WeatherAction.TOKEN_END));
				}
			}
					
					
					
		
		
		}
		if(!loadTrainingData(trainingData.size())){
			System.out.print("Create training data...");
			Object[] results = inferFeatureAndCostVectors();
            System.out.print("almost...");
		}
		
		
		
		
		
		
	}
	@SuppressWarnings("unchecked")
	public boolean loadTrainingData(int dataSize){
		String file1 = "cache/attrTrainingData" + "_" + dataSize;
        String file2 = "cache/wordTrainingData" + "_" + dataSize;
        String file3 = "cache/fieldTrainingData" + "_" + dataSize;
        FileInputStream fin1 = null;
        ObjectInputStream ois1 = null;
        FileInputStream fin2 = null;
        ObjectInputStream ois2 = null;
        FileInputStream fin3 = null;
        ObjectInputStream ois3 = null;
        if ((new File(file1)).exists()
                && (new File(file2)).exists()
                &&(new File(file3)).exists()) {
        	try{
        		System.out.println("loading training data");
        		fin1 = new FileInputStream(file1);
                ois1 = new ObjectInputStream(fin1);
                Object o1 = ois1.readObject();
                if(getAttributeTrainingData()==null){
                	if(o1 instanceof ArrayList){
                		setAttributeTrainingData(new ArrayList<>((Collection<? extends Instance>) o1));
                	}
                }else if(o1 instanceof ArrayList){
                	getAttributeTrainingData().addAll(new ArrayList<>((Collection<? extends Instance>) o1));
                }
                fin2 = new FileInputStream(file2);
                ois2 = new ObjectInputStream(fin2);
                Object o2 = ois2.readObject();
                if(getWordTrainingData()==null){
                	if(o2 instanceof HashMap){
                		setWordTrainingData(new HashMap<>( (Map<? extends String, ? extends HashMap<String, ArrayList<Instance>>>)o2));
                	}
                }else if(o2 instanceof HashMap){
                	getWordTrainingData().putAll(new HashMap<>( (Map<? extends String, ? extends HashMap<String, ArrayList<Instance>>>)o2));
                }
                fin3 = new FileInputStream(file3);
                ois3 = new ObjectInputStream(fin3);
                Object o3 = ois3.readObject();
                if(getFieldTrainingData()==null){
                	if(o3 instanceof HashMap){
                		setFieldTrainingData(new HashMap<>((Map<? extends String, ? extends ArrayList<Instance>>)o3));
                	}
                	
                }else if(o3 instanceof HashMap){
                	getFieldTrainingData().putAll(new HashMap<>((Map<? extends String, ? extends ArrayList<Instance>>)o3));
                }
                
                
        		
        	}catch(Exception ex){
        		
        	}finally{
        		try {
                    fin1.close();
                    fin2.close();
                    fin3.close();
                } catch (IOException ex) {
                }
                try {
                    ois1.close();
                    ois2.close();
                    ois3.close();
                } catch (IOException ex) {
                }
        		
        	}
        }else{
        	return false;
        }
        return true;

	}
	public void writeTrainingData(int dataSize){
		String file1 = "cache/attrTrainingData" + "_" + dataSize;
        String file2 = "cache/wordTrainingData" + "_" + dataSize;
        String file3 = "cache/fieldTrainingData" + "_" + dataSize;
        FileOutputStream fout1 = null;
        ObjectOutputStream oos1 = null;
        FileOutputStream fout2 = null;
        ObjectOutputStream oos2 = null;
        FileOutputStream fout3 = null;
        ObjectOutputStream oos3 = null;
        try {
            System.out.print("Write Training Data...");
            fout1 = new FileOutputStream(file1);
            oos1 = new ObjectOutputStream(fout1);
            oos1.writeObject(getAttributeTrainingData());

            fout2 = new FileOutputStream(file2);
            oos2 = new ObjectOutputStream(fout2);
            oos2.writeObject(getWordTrainingData());
            
            fout3 = new FileOutputStream(file3);
            oos3 = new ObjectOutputStream(fout3);
            oos3.writeObject(getFieldTrainingData());
            
            

        } catch (IOException ex) {
        } finally {
            try {
                fout1.close();
                fout2.close();
                fout3.close();
            } catch (IOException ex) {
            }
            try {
                oos1.close();
                oos2.close();
                oos3.close();
            } catch (IOException ex) {
            }
        }
        
        
		
	}
	public void writeLMs() {
        String file2 = "cache/wordLMs" ;
        String file3 = "cache/attrLMs"  ;
        String file1 = "cache/fieldLMs";
        FileOutputStream fout1 = null;
        ObjectOutputStream oos1 = null;
        FileOutputStream fout2 = null;
        ObjectOutputStream oos2 = null;
        FileOutputStream fout3 = null;
        ObjectOutputStream oos3 = null;
        try {
            System.out.print("Write LMs...");
            fout1 = new FileOutputStream(file1);
            oos1 = new ObjectOutputStream(fout1);
            oos1.writeObject(fieldLMs);
            
            fout2 = new FileOutputStream(file2);
            oos2 = new ObjectOutputStream(fout2);
            oos2.writeObject(wordLMs);

            fout3 = new FileOutputStream(file3);
            oos3 = new ObjectOutputStream(fout3);
            oos3.writeObject(attrLMs);
        } catch (IOException ex) {
        } finally {
            try {
            	fout1.close();
                fout2.close();
                fout3.close();
            } catch (IOException ex) {
            }
            try {
            	oos1.close();
                oos2.close();
                oos3.close();
            } catch (IOException ex) {
            }
            
        }
    }
	public boolean loadLMs() {
		String file1 = "cache/fieldLMs" ;
        String file2 = "cache/wordLMs" ;
        String file3 = "cache/attrLMs";
        FileInputStream fin1 = null;
        ObjectInputStream ois1 = null;
        FileInputStream fin2 = null;
        ObjectInputStream ois2 = null;
        FileInputStream fin3 = null;
        ObjectInputStream ois3 = null;
        if ((new File(file2)).exists()
                && (new File(file3)).exists()
                &&(new File(file1)).exists()) {
            try {
                System.out.print("Load language models...");
                fin1 = new FileInputStream(file1);
                ois1 = new ObjectInputStream(fin1);
                Object o1 = ois1.readObject();
                
                if (o1 instanceof WeatherSimpleLM) {
                	fieldLMs = (WeatherSimpleLM) o1;
                }
                
                fin2 = new FileInputStream(file2);
                ois2 = new ObjectInputStream(fin2);
                Object o2 = ois2.readObject();
                
                if (o2 instanceof WeatherSimpleLM) {
                	wordLMs = (WeatherSimpleLM) o2;
                }
               

                fin3 = new FileInputStream(file3);
                ois3 = new ObjectInputStream(fin3);
                Object o3 = ois3.readObject();
                if(o3 instanceof WeatherSimpleLM){
                	attrLMs = (WeatherSimpleLM) o3;
                }

            } catch (ClassNotFoundException | IOException ex) {
            } finally {
                try {
                    fin2.close();
                    fin3.close();
                } catch (IOException ex) {
                }
                try {
                    ois2.close();
                    ois3.close();
                } catch (IOException ex) {
                }
            }
        } else {
            return false;
        }
        return true;
    }
	
	public void createNaiveAlignments(ArrayList<WeatherDatasetInstance> trainingData){
		HashMap<WeatherDatasetInstance,ArrayList<WeatherAction>> punctRealizations = new HashMap<>();
		HashMap<ArrayList<WeatherAction>,HashMap<WeatherAction,Integer>> punctPatterns = new HashMap<>();
		System.out.println("create Naive Alignments");
		trainingData.stream().map(di->{
			
			HashMap<ArrayList<WeatherAction>, ArrayList<WeatherAction>> calculatedRealizationsCache = new HashMap<>();//key directReferenceSequence
            HashSet<ArrayList<WeatherAction>> initRealizations = new HashSet<>();
            if (!calculatedRealizationsCache.containsKey(di.getDirectReferenceSequence())) {
                initRealizations.add(di.getDirectReferenceSequence());
            }
            initRealizations.stream().map((realization) -> {
            	/*
            	for(int i=0;i<realization.size();i++){
            		System.out.print(realization.get(i).getWord());
            	}
            	System.out.println("\n");*/
            	HashMap<String, HashMap<String,String>> values = new HashMap<>();
            	values.putAll(di.getMR().getAttrFieldValue());
            	ArrayList<WeatherAction> randomRealization = new ArrayList<>();
            	for(int i=0;i<realization.size();i++){
            		WeatherAction a = realization.get(i);
            		
            		if(a.getWord().equals(WeatherAction.TOKEN_PUNCT)||a.getWord().contains("X")){
            			
            			randomRealization.add(new WeatherAction(a.getWord(),a.getField(),a.getField()));
            		}else{
            			randomRealization.add(new WeatherAction(a.getWord(),"",""));
            		}
            	}
            	
            	//indexAlignments
            	
            	HashMap<Double, HashMap<String, ArrayList<Integer>>> indexAlignments = new HashMap<>();
            	values.keySet().forEach(attr->{
            		values.get(attr).keySet().forEach(field->{
            			String value = values.get(attr).get(field);
            			if(!Pattern.matches("([0-9]+)", value)&&!
                				(field.equals("time")||field.equals("max")||field.equals("mean")||field.equals("min")
                						||Pattern.matches("([0-9]+)-([0-9]+)",value ))&&valueAlignments.containsKey(value)
                				&&!value.contains("X")){
            				String valueToCheck = value;
            				if(valueToCheck.equals("windDir")){
            					valueToCheck = "wind Dir";
                    		}
                    		if(valueToCheck.equals("windSpeed")){
                    			valueToCheck = "wind Speed";
                    		}
                    		if(valueToCheck.equals("windChill")){
                    			valueToCheck = "wind Chill";
                    		}
                    		if(valueToCheck.equals("freezingRainChance")){
                    			valueToCheck = "freezing Rain Chance";
                    		}
                    		if(valueToCheck.equals("freezingRainChance")){
                    			valueToCheck = "freezing Rain Chance";
                    		}
                    		if(valueToCheck.equals("precipPotential")){
                    			valueToCheck = "precip Potential";
                    		}
                    		if(valueToCheck.equals("rainChance")){
                    			valueToCheck = "rain Chance";
                    		}
                    		if(valueToCheck.equals("thunderChance")){
                    			valueToCheck = "thunder Chance";
                    		}
                    		if(valueToCheck.equals("sleetChance")){
                    			valueToCheck = "sleet Chance";
                    		}
                    		if(valueToCheck.equals("snowChance")){
                    			valueToCheck = "snow Chance";
                    		}
                    		if(valueToCheck.equals("skyCover")){
                    			valueToCheck = "sky Cover";
                    		}
                    		for(ArrayList<String> align : valueAlignments.get(valueToCheck).keySet() ){
                    			int n = align.size();
                    			for (int i = 0; i <= randomRealization.size() - n; i++) {
                                    ArrayList<String> compare = new ArrayList<String>();
                                    ArrayList<Integer> indexAlignment = new ArrayList<Integer>();
                                    for (int j = 0; j < n; j++) {
                                        compare.add(randomRealization.get(i + j).getWord());
                                        indexAlignment.add(i + j);
                                    }
                                    if (compare.equals(align)) {
                                        if (!indexAlignments.containsKey(valueAlignments.get(valueToCheck).get(align))) {
                                            indexAlignments.put(valueAlignments.get(valueToCheck).get(align), new HashMap<>());
                                        }
                                        indexAlignments.get(valueAlignments.get(valueToCheck).get(align)).put(attr +"="+field+"=" + valueToCheck, indexAlignment);
                                    }
                                }
                    			
                    		}
            				
            			}
            		});
            	});
            	ArrayList<Double> similarities = new ArrayList<>(indexAlignments.keySet());
            	Collections.sort(similarities);
            	HashSet<String> assignedAttrFieldValues = new HashSet<String>();
                HashSet<Integer> assignedIntegers = new HashSet<Integer>();
                for(int i = similarities.size()-1;i>=0;i--){
                	for(String attrFieldValue : indexAlignments.get(similarities.get(i)).keySet()){
                		if(!assignedAttrFieldValues.contains(attrFieldValue)){
                			boolean isUnassigned = true;
                			for (Integer index : indexAlignments.get(similarities.get(i)).get(attrFieldValue)) {
                                if (assignedIntegers.contains(index)) {
                                    isUnassigned = false;
                                }
                            }
                			if (isUnassigned) {
                				assignedAttrFieldValues.add(attrFieldValue);
                                for (Integer index : indexAlignments.get(similarities.get(i)).get(attrFieldValue)) {
                                    assignedIntegers.add(index);
                                    randomRealization.get(index).setAttribute(attrFieldValue.toLowerCase().trim());
                                    randomRealization.get(index).setField(attrFieldValue.toLowerCase().trim());
                                    
                                }
                            }
                		}
                	}
                }/*
                for(int i=0;i<randomRealization.size();i++){
            		System.out.print(randomRealization.get(i).getWord()+" ");
            	}
            	System.out.println("\n");*/
            /*   
            for(int i=0;i<randomRealization.size();i++){
            	System.out.print(randomRealization.get(i).getWord()+" ");
            } 
            System.out.println("\n");
            for(int i=0;i<randomRealization.size();i++){
            	if(randomRealization.get(i).getField().isEmpty()){
            		System.out.print(" ");
            	}else{
            	System.out.print(randomRealization.get(i).getField()+" ");
            	}
            } 
            System.out.println("\n");
            for(int i=0;i<randomRealization.size();i++){
            	if(randomRealization.get(i).getAttribute().isEmpty()){
            		System.out.print(" ");
            	}else{
            	System.out.print(randomRealization.get(i).getAttribute()+" ");
            	}
            } 
            System.out.println("\n");
            */
            
            //after check, no randomReealization is empty.
            //finish align the valueAlignments 
            //next to randomly split from the middle of 2 besides attribute
                
                String previousAttr = "";
                String previousField = "";
                
                int start = -1;
                for (int i = 0; i < randomRealization.size(); i++) {
                    if (!randomRealization.get(i).getAttribute().equals(WeatherAction.TOKEN_PUNCT)
                            && !randomRealization.get(i).getAttribute().isEmpty()
                            && !randomRealization.get(i).getAttribute().equals("[]")
                            && !randomRealization.get(i).getField().isEmpty()
                            &&!randomRealization.get(i).getField().equals(WeatherAction.TOKEN_PUNCT)
                            &&!randomRealization.get(i).getField().equals("[]")) {
                        if (start != -1) {
                            int middle = (start + i - 1) / 2 + 1;
                            for (int j = start; j < middle; j++) {
                                if (randomRealization.get(j).getAttribute().isEmpty()
                                        || randomRealization.get(j).getAttribute().equals("[]")
                                        ||randomRealization.get(j).getField().isEmpty()
                                       ||randomRealization.get(j).getField().equals("[]") ) {
                                    randomRealization.get(j).setAttribute(previousAttr);
                                    randomRealization.get(j).setField(previousField);
                                }
                            }
                            for (int j = middle; j < i; j++) {
                                if (randomRealization.get(j).getAttribute().isEmpty()
                                        || randomRealization.get(j).getAttribute().equals("[]")
                                        ||randomRealization.get(j).getField().isEmpty()
                                        ||randomRealization.get(j).getField().equals("[]") ) {
                                    randomRealization.get(j).setAttribute(randomRealization.get(i).getAttribute());
                                    randomRealization.get(j).setField(randomRealization.get(i).getField());
                                }
                            }
                        }
                        start = i;
                        previousAttr = randomRealization.get(i).getAttribute();
                        previousField = randomRealization.get(i).getField();
                    } else {
                        previousAttr = "";
                        previousField = "";
                    }
                }
                for(int i=0;i<randomRealization.size();i++){
                	
                	if(randomRealization.get(i).getAttribute().isEmpty()||randomRealization.get(i).getField().isEmpty()
                			||randomRealization.get(i).getAttribute().equals("[]")||randomRealization.get(i).getField().equals("[]")){
                		boolean find = false;
                		int n = i;
                		int m = i;
                		while(!find){
                			n = n+1;
                			m = m-1;
                			if(n>=randomRealization.size()){
                				n=randomRealization.size()-1;
                			}
                			if(m<=0){
                				m = 0;
                			}
                			if(!randomRealization.get(n).getAttribute().isEmpty()&&
                					!randomRealization.get(n).getAttribute().equals("[]")&&
                					!randomRealization.get(n).getField().equals("[]")&&
                					!randomRealization.get(n).getField().isEmpty()&&
                					!randomRealization.get(n).getAttribute().equals(WeatherAction.TOKEN_PUNCT)&&
                					!randomRealization.get(n).getField().equals(WeatherAction.TOKEN_PUNCT)
                					){
                				find = true;
                				randomRealization.get(i).setAttribute(randomRealization.get(n).getAttribute());
                				randomRealization.get(i).setField(randomRealization.get(n).getField());
                			}else if(!randomRealization.get(m).getAttribute().isEmpty()&&
                					!randomRealization.get(m).getAttribute().equals("[]")&&
                					!randomRealization.get(m).getField().equals("[]")&&
                					!randomRealization.get(m).getField().isEmpty()&&
                					!randomRealization.get(m).getAttribute().equals(WeatherAction.TOKEN_PUNCT)&&
                					!randomRealization.get(m).getField().equals(WeatherAction.TOKEN_PUNCT)){
                				find = true;
                				randomRealization.get(i).setAttribute(randomRealization.get(m).getAttribute());
                				randomRealization.get(i).setField(randomRealization.get(m).getField());
                			}
                		}
                	}
                }
                
                
                /*
                //backwards
                
                previousAttr = "";
                previousField = "";
                for(int i=randomRealization.size()-1;i>=0;i--){
                	if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                        if (!previousAttr.isEmpty()&&!previousField.isEmpty()) {
                            randomRealization.get(i).setAttribute(previousAttr);
                            randomRealization.get(i).setField(previousField);
                        }
                	}else if (!randomRealization.get(i).getAttribute().equals(WeatherAction.TOKEN_PUNCT)&&
                			!randomRealization.get(i).getField().equals(WeatherAction.TOKEN_PUNCT)) {
                        previousAttr = randomRealization.get(i).getAttribute();
                        previousField = randomRealization.get(i).getField();
                    } else {
                        previousAttr = "";
                        previousField = "";
                    }
                }
                //forwards
                previousAttr = "";
                previousField = "";
                for(int i=0;i<randomRealization.size();i++){
                	if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                        if (!previousAttr.isEmpty()&&!previousField.isEmpty()) {
                            randomRealization.get(i).setAttribute(previousAttr);
                            randomRealization.get(i).setField(previousField);
                        }
                	}else if (!randomRealization.get(i).getAttribute().equals(WeatherAction.TOKEN_PUNCT)&&
                			!randomRealization.get(i).getField().equals(WeatherAction.TOKEN_PUNCT)) {
                        previousAttr = randomRealization.get(i).getAttribute();
                        previousField = randomRealization.get(i).getField();
                    } else {
                        previousAttr = "";
                        previousField = "";
                    }
                }
                //backwards
                
                previousAttr = "";
                previousField = "";
                for(int i=randomRealization.size()-1;i>=0;i--){
                	if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                        if (!previousAttr.isEmpty()&&!previousField.isEmpty()) {
                            randomRealization.get(i).setAttribute(previousAttr);
                            randomRealization.get(i).setField(previousField);
                        }
                	}else if (!randomRealization.get(i).getAttribute().equals(WeatherAction.TOKEN_PUNCT)&&
                			!randomRealization.get(i).getField().equals(WeatherAction.TOKEN_PUNCT)) {
                        previousAttr = randomRealization.get(i).getAttribute();
                        previousField = randomRealization.get(i).getField();
                    } else {
                        previousAttr = "";
                        previousField = "";
                    }
                }
                //forwards
                previousAttr = "";
                previousField = "";
                for(int i=0;i<randomRealization.size();i++){
                	if (randomRealization.get(i).getAttribute().isEmpty() || randomRealization.get(i).getAttribute().equals("[]")) {
                        if (!previousAttr.isEmpty()&&!previousField.isEmpty()) {
                            randomRealization.get(i).setAttribute(previousAttr);
                            randomRealization.get(i).setField(previousField);
                        }
                	}else if (!randomRealization.get(i).getAttribute().equals(WeatherAction.TOKEN_PUNCT)&&
                			!randomRealization.get(i).getField().equals(WeatherAction.TOKEN_PUNCT)) {
                        previousAttr = randomRealization.get(i).getAttribute();
                        previousField = randomRealization.get(i).getField();
                    } else {
                        previousAttr = "";
                        previousField = "";
                    }
                }*/
                //check
                //for(int i=0;i<randomRealization.size();i++){
                	//System.out.print(randomRealization.get(i).getWord()+" ");
                //} 
                //System.out.println("\n");
                /*
                for(int i=0;i<randomRealization.size();i++){
                	if(randomRealization.get(i).getField().isEmpty()||randomRealization.get(i).getField().equals("[]")){
                		System.out.println("@empty@ ");
                	}else{
                	//System.out.print(randomRealization.get(i).getField()+" ");
                	}
                } 
                //System.out.println("\n");
                for(int i=0;i<randomRealization.size();i++){
                	if(randomRealization.get(i).getAttribute().isEmpty()||randomRealization.get(i).getAttribute().equals("[]")){
                		System.out.println("@empty@ ");
                	}else{
                	//System.out.print(randomRealization.get(i).getAttribute()+" ");
                	}
                } */
                //System.out.println("\n");
                
                //filter out punctuation
                //ArrayList<WeatherAction> cleanRandomRealization = new ArrayList<>();
                //randomRealization.stream().filter((a)->(!a.getWord().equals(WeatherAction.TOKEN_PUNCT))).forEachOrdered(a->{
                	//cleanRandomRealization.add(a);
                //});
                
                //add end token
                ArrayList<WeatherAction> endAttrRealization = new ArrayList<>();
                ArrayList<WeatherAction> endFieldRealization = new ArrayList<>();
                previousAttr = randomRealization.get(0).getAttribute();
                previousField = randomRealization.get(0).getField();
                for(int i=0;i<randomRealization.size();i++){
                	WeatherAction a  = randomRealization.get(i);
                	
                	if(!a.getAttribute().equals(WeatherAction.TOKEN_PUNCT)){
                		
                		if(!a.getAttribute().isEmpty()
                				&&!a.getField().isEmpty()
                				&&!a.getAttribute().equals(previousAttr)){
                			if(randomRealization.get(i-1).getAttribute().equals(WeatherAction.TOKEN_PUNCT)
                					){
                			//endAttrRealization.add(new WeatherAction(WeatherAction.TOKEN_END,previousField,previousAttr));
                			endAttrRealization.add(a);
                			}else{
                				endAttrRealization.add(new WeatherAction(WeatherAction.TOKEN_END,previousField,previousAttr));
                    			endAttrRealization.add(a);
                			}
                			
                		}else{
                			endAttrRealization.add(a);
                		}
                	}else{
                		if(i!=randomRealization.size()-1){
                			if(!randomRealization.get(i-1).getAttribute().equals(randomRealization.get(i+1).getAttribute())){
                				endAttrRealization.add(new WeatherAction(WeatherAction.TOKEN_END,previousField,previousAttr));
                    			endAttrRealization.add(a);
                			}
                		}
                		else{
                			endAttrRealization.add(new WeatherAction(WeatherAction.TOKEN_END,previousField,previousAttr));
                			endAttrRealization.add(a);
                			
                		}
                	}
                	//endAttrRealization.add(a);
                		
                	if(!a.getAttribute().equals(WeatherAction.TOKEN_PUNCT)){
                		if(!a.getAttribute().isEmpty()
                				&&!a.getField().isEmpty()
                				&&(!a.getField().equals(previousField)
                				||!a.getAttribute().equals(previousAttr))){
                			endFieldRealization.add(new WeatherAction(WeatherAction.TOKEN_END,previousField,previousAttr));
                		
                        
                		}
                	//endFieldRealization.add(a);
                		previousAttr = a.getAttribute();
                		previousField = a.getField();
                	}
                }
                //get maxFieldLength and maxAttributeLength with end token added
                //endAttrRealization.add(new WeatherAction(WeatherAction.TOKEN_END,previousField,previousAttr));
                endFieldRealization.add(new WeatherAction(WeatherAction.TOKEN_END,previousField,previousAttr));
                endAttrRealization.add(new WeatherAction(WeatherAction.TOKEN_END,WeatherAction.TOKEN_END,WeatherAction.TOKEN_END));
                endFieldRealization.add(new WeatherAction(WeatherAction.TOKEN_END,WeatherAction.TOKEN_END,WeatherAction.TOKEN_END));
                calculatedRealizationsCache.put(realization, endAttrRealization);
                /*
                ArrayList<String> attrValues = new ArrayList<String>();
                ArrayList<String> fieldValues = new ArrayList<String>();
                endAttrRealization.forEach((a) -> {
                	if(!a.getAttribute().equals(WeatherAction.TOKEN_PUNCT)){
                    if (attrValues.isEmpty()) {
                        attrValues.add(a.getAttribute());
                    } else if (!attrValues.get(attrValues.size() - 1).equals(a.getAttribute())) {
                        attrValues.add(a.getAttribute());
                    }
                	}
                });
                endAttrRealization.forEach((a) -> {
                    if (fieldValues.isEmpty()) {
                    	fieldValues.add(a.getField());
                    } else if (!fieldValues.get(fieldValues.size() - 1).equals(a.getField())) {
                    	fieldValues.add(a.getField());
                    }
                });
                if (attrValues.size() > maxAttributeSequenceLength) {
                	maxAttributeSequenceLength=attrValues.size();
                }
                if (endFieldRealization.size() > maxFieldSequenceLength) {
                	maxFieldSequenceLength=endFieldRealization.size();
                }*//*
                for(int i=0;i<endAttrRealization.size();i++){
                	System.out.print(endAttrRealization.get(i).getAttribute()+"=="+endAttrRealization.get(i).getWord()+"   ");
                }
                System.out.println("\n");*/
                ArrayList<WeatherAction> punctRealization = new ArrayList<>();
                punctRealization.addAll(endAttrRealization);/*
                previousAttr = "";
                previousField = "";
                for(int i=0;i<punctRealization.size();i++){
                	if(!punctRealization.get(i).getAttribute().equals(WeatherAction.TOKEN_PUNCT)){
                		if(!punctRealization.get(i).getAttribute().equals(previousAttr)
                                && !previousAttr.isEmpty()){
                			punctRealization.add(new WeatherAction(WeatherAction.TOKEN_END,previousField,previousAttr));
                		}
                		previousAttr = punctRealization.get(i).getAttribute();
                		previousField = punctRealization.get(i).getField();
                	}
                }
                if(!punctRealization.get(punctRealization.size()-1).getWord().equals(WeatherAction.TOKEN_END)){
                	punctRealization.add(new WeatherAction(WeatherAction.TOKEN_END,previousField,previousAttr));
                }*/
                
            	return punctRealization;
            }).map(punctRealization->{
            	punctRealizations.put(di, punctRealization);
            	return punctRealization;
            }).forEachOrdered((punctRealization)->{
            	for(int i=0;i<punctRealization.size();i++){
            		if(punctRealization.get(i).getAttribute().equals(WeatherAction.TOKEN_PUNCT)){
            			WeatherAction a = punctRealization.get(i);
            			ArrayList<WeatherAction> surroundingActions = new ArrayList<>();
            			if(i==1){
            				surroundingActions.add(punctRealization.get(0));
            				surroundingActions.add(punctRealization.get(2));
            				surroundingActions.add(punctRealization.get(3));
            				
            			}
            			else if(i==punctRealization.size()-2){
            				surroundingActions.add(punctRealization.get(punctRealization.size()-3));
            				surroundingActions.add(punctRealization.get(punctRealization.size()-4));
            				surroundingActions.add(punctRealization.get(punctRealization.size()-1));
            				
            			}else{
            				surroundingActions.add(punctRealization.get(i-2));
            				surroundingActions.add(punctRealization.get(i-1));
            				surroundingActions.add(punctRealization.get(i+2));
            				surroundingActions.add(punctRealization.get(i+1));
            			}
            			if(!punctPatterns.containsKey(surroundingActions)){
            				punctPatterns.put(surroundingActions, new HashMap<>());
            			}
            			if(!punctPatterns.get(surroundingActions).containsKey(a)){
            				punctPatterns.get(surroundingActions).put(a, 1);
            			}else{
            				punctPatterns.get(surroundingActions).put(a, punctPatterns.get(surroundingActions).get(a)+1);
            			}
            		}
            		
            	}
            });
            di.setDirectReferenceSequence(calculatedRealizationsCache.get(di.getDirectReferenceSequence()));
			return di;
		}).forEachOrdered(di->{
			
		});
		
		/*
		punctPatterns.keySet().stream().forEach(surrounds->{
			punctRealizations.keySet().stream().forEach(di->{
				ArrayList<WeatherAction> punctRealization = punctRealizations.get(di) ;
			int s = surrounds.size();
				
			for(int i=0;i<=punctRealization.size()-s;i++){
				boolean find = true;
				int j = 0;
				while(j<s&&find){
					if(!punctRealization.get(i+j).getWord().equals(surrounds.get(j).getWord())){
						find = false;
					}
					j++;
				}
				if(find){
					WeatherAction a = new WeatherAction("","","");
					if(!punctPatterns.get(surrounds).containsKey(a)){
            			punctPatterns.get(surrounds).put(a, 1);
            		}else{
            			punctPatterns.get(surrounds).put(a, punctPatterns.get(surrounds).get(a)+1);
            		}
				}
			}
			WeatherAction bestAction = null;
			int bestCount = 0;
			for(WeatherAction a :punctPatterns.get(surrounds).keySet()){
				if(punctPatterns.get(surrounds).get(a)>bestCount){
					bestCount = punctPatterns.get(surrounds).get(a);
					bestAction = a;
				}
			}
			if(!bestAction.getWord().isEmpty()){
				punctuationPatterns.put(surrounds, bestAction);
			}
			
		});
	});*/
		
		
		System.out.println("finish!");
	}
	public Instance createAttributeInstance(String bestAction,ArrayList<String> previousGeneratedAttrs
			,HashSet<String> attrsAlreadyMentioned,HashSet<String> attrsToBeMentioned
			,WeatherMeaningRepresentation MR, HashSet<String> availableAttr){
		TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
		
		if(!bestAction.isEmpty()){
			//COSTS
			if(bestAction.equals(WeatherAction.TOKEN_END)){
				costs.put(WeatherAction.TOKEN_END, 0.0);
				availableAttr.stream().forEach(attribute->{
					costs.put(attribute, 1.0);
				});
			}else{
				costs.put(WeatherAction.TOKEN_END, 1.0);
				availableAttr.stream().forEach(attribute->{
					if(attributeFields.equals(bestAction.split("=")[0])){
						costs.put(attribute, 0.0);
					}else{
						costs.put(attribute, 1.0);
					}
				});
				
			}
		}
		
		return createContentInstanceWithCosts( costs, previousGeneratedAttrs
				, attrsAlreadyMentioned, attrsToBeMentioned
				, availableAttr, MR);
		
		
	}
	public Instance createContentInstanceWithCosts(TObjectDoubleHashMap<String> costs,ArrayList<String> previousGeneratedAttrs
			,HashSet<String> attrsAlreadyMentioned,HashSet<String> attrsToBeMentioned
			,HashSet<String> availableAttr,WeatherMeaningRepresentation MR){
		TObjectDoubleHashMap<String> generalFeatures = new TObjectDoubleHashMap<>();
        HashMap<String, TObjectDoubleHashMap<String>> valueSpecificFeatures = new HashMap<>();
        availableAttr.stream().forEach(attribute->{
        	valueSpecificFeatures.put(attribute, new TObjectDoubleHashMap<String>());
        });
        //previous generated attrs
        ArrayList<String> mentionedAttr = new ArrayList<>();
        previousGeneratedAttrs.stream().filter((attrFieldValue)->(!attrFieldValue.equals(WeatherAction.TOKEN_END)&&
        		!attrFieldValue.equals(WeatherAction.TOKEN_START))).forEach((attribute)->{
        			String attr = attribute;
        			if(attribute.contains("=")){
        				 attr = attribute.split("=")[0];
        			}
        			
        			if(!mentionedAttr.get(mentionedAttr.size()-1).equals(attr)){
        				mentionedAttr.add(attr);
        			}
        		});
        
        //general features
        for (int j = 1; j <= 1; j++) {
            String previousAttr = "@@";
            if (mentionedAttr.size() - j >= 0) {
            	previousAttr = mentionedAttr.get(mentionedAttr.size() - j).trim();
            }
            generalFeatures.put("feature_attr_" + j + "_" + previousAttr, 1.0);
        }
        //previous attribute N-Grams
        String previousAttr = "@@";
        String prevTimeValue = "";
        if (mentionedAttr.size() - 1 >= 0) {
        	previousAttr = mentionedAttr.get(mentionedAttr.size() - 1).trim();
        	if(!previousAttr.equals(WeatherAction.TOKEN_END)&&MR.getAttrFieldValue().containsKey(previousAttr)){
        		for(String field : MR.getAttrFieldValue().get(previousAttr).keySet()){
        			if(field.contains("time")){
        				prevTimeValue = MR.getAttrFieldValue().get(previousAttr).get(field);
        			}
        		}
        	}
        	
        }
        String previous2Attr = "@@";
        String prev2TimeValue = "";
        if (mentionedAttr.size() - 2 >= 0) {
        	previous2Attr = mentionedAttr.get(mentionedAttr.size() - 2).trim();
        	if(!previous2Attr.equals(WeatherAction.TOKEN_END)&&MR.getAttrFieldValue().containsKey(previous2Attr)){
        		for(String field: MR.getAttrFieldValue().get(previous2Attr).keySet()){
        			if(field.contains("time")){
        				prev2TimeValue = MR.getAttrFieldValue().get(previous2Attr).get(field);
        			}
        		}
        	}
        }
        String previous3Attr = "@@";
        String prev3TimeValue = "";
        if (mentionedAttr.size() - 3 >= 0) {
        	previous3Attr = mentionedAttr.get(mentionedAttr.size() - 3).trim();
        	if(!previous3Attr.equals(WeatherAction.TOKEN_END)&&MR.getAttrFieldValue().containsKey(previous3Attr)){
        		for(String field: MR.getAttrFieldValue().get(previous3Attr).keySet()){
        			if(field.contains("time")){
        				prev3TimeValue  = MR.getAttrFieldValue().get(previous3Attr).get(field);
        			}
        		}
        	}
        }
        String previous4Attr = "@@";
        String prev4TimeValue = "";
        if (mentionedAttr.size() - 4 >= 0) {
        	previous4Attr = mentionedAttr.get(mentionedAttr.size() - 4).trim();
        	if(!previous4Attr.equals(WeatherAction.TOKEN_END)&&MR.getAttrFieldValue().containsKey(previous4Attr)){
        		for(String field : MR.getAttrFieldValue().get(previous4Attr).keySet()){
        			if(field.contains("time")){
        				prev4TimeValue = MR.getAttrFieldValue().get(previous4Attr).get(field);
        			}
        		}
        	}
        }
        String previous5Attr = "@@";
        String prev5TimeValue = "";
        if (mentionedAttr.size() - 5 >= 0) {
        	previous5Attr = mentionedAttr.get(mentionedAttr.size() - 5).trim();
        	if(!previous5Attr.equals(WeatherAction.TOKEN_END)&&MR.getAttrFieldValue().containsKey(previous5Attr)){
        		for(String field : MR.getAttrFieldValue().get(previous5Attr).keySet()){
        			if(field.contains("time")){
        				prev5TimeValue = MR.getAttrFieldValue().get(previous5Attr).get(field);
        			}
        		}
        	}
        }
        String previousBigramAttr = previous2Attr + "|" + previousAttr;
        String previousTrigramAttr = previous3Attr + "|" + previous2Attr + "|" + previousAttr;
        String previous4gramAttr = previous4Attr + "|" + previous3Attr + "|" + previous2Attr + "|" + previousAttr;
        String previous5gramAttr = previous5Attr + "|" + previous4Attr + "|" + previous3Attr + "|" + previous2Attr + "|" + previousAttr;
        generalFeatures.put("feature_attr_bigram_" + previousBigramAttr, 1.0);
        generalFeatures.put("feature_attr_trigram_" + previousTrigramAttr, 1.0);
        generalFeatures.put("feature_attr_4gram_" + previous4gramAttr, 1.0);
        generalFeatures.put("feature_attr_5gram_" + previous5gramAttr, 1.0);
        
        //If arguments have been generated or not
        for (int i = 0; i < mentionedAttr.size(); i++) {
            generalFeatures.put("feature_attr_allreadyMentioned_" + mentionedAttr.get(i), 1.0);
        }
        //If arguments should still be generated or not
        attrsToBeMentioned.forEach((attr) -> {
            generalFeatures.put("feature_attr_toBeMentioned_" + attr, 1.0);
        }); 
        //Which attrs are in the MR and which are not
        
        availableAttr.forEach((attribute) -> {
            if (MR.getAttrFieldValue().keySet().contains(attribute)) {
                generalFeatures.put("feature_attr_inMR_" + attribute, 1.0);
            } else {
                generalFeatures.put("feature_attr_notInMR_" + attribute, 1.0);
            }
        });
        //HashSet<String> attrsToBeMentioned = new HashSet<>();
        /*
        attrsToBeMentioned.stream().forEach((attr)->{
        	String attr = attrFieldValue.split("=")[0];
        	attrsToBeMentioned.add(attr);
        });*/
        //just attribute
        /*
        ArrayList<String> mentionedAttrs = new ArrayList<>();
        for(int i=0; i< mentionedAttrFieldValues.size();i++){
        	String attr = mentionedAttrFieldValues.get(i).split("=")[0];
        	mentionedAttrs.add(attr);
        }
        HashSet<String> attrsToBeMentioned = new HashSet<>();
        attrFieldValuesToBeMentioned.stream().forEach((attrFieldValue)->{
        	String attr = attrFieldValue.split("=")[0];
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
        mentionedAttrs.forEach((attr) -> {
            generalFeatures.put("feature_attr_alreadyMentioned_" + attr, 1.0);
        });
        //If arguments should still be generated or not
        attrsToBeMentioned.forEach((attr) -> {
            generalFeatures.put("feature_attr_toBeMentioned_" + attr, 1.0);
        });*/
        //value specific features
        for(String attribute : availableAttr){
        	if(attribute.equals(WeatherAction.TOKEN_END)){
        		// check if attribute not mentioned when going to end, at the end of attribute action sequence
        		if (attrsToBeMentioned.isEmpty()) {
                    valueSpecificFeatures.get(attribute).put("global_feature_specific_allAttrsMentioned", 1.0);
                } else {
                    valueSpecificFeatures.get(attribute).put("global_feature_specific_allAttrsNotMentioned", 1.0);
                }
        		
        	}else{
        		
        		//Is attr in MR?
        		if (MR.getAttrFieldValue().get(attribute) != null) {
                    valueSpecificFeatures.get(attribute).put("global_feature_specific_isInMR", 1.0);
                    String currentTimeValue = "";
                    for(String field: MR.getAttrFieldValue().get(attribute).keySet()){
                    	if(field.contains("time")){
                    		currentTimeValue = MR.getAttrFieldValue().get(attribute).get(field);
                    		
                    	}
                    }
                    if(!currentTimeValue.isEmpty()&&!prevTimeValue.isEmpty()){
                    	if(currentTimeValue.equals(prevTimeValue)){
                    		valueSpecificFeatures.get(attribute).put("global_feature_specific_equal_to_prev1TimeValue", 1.0);
                    	}
                    	else{
                    		valueSpecificFeatures.get(attribute).put("global_feature_specific_Not_equal_to_prev1TimeValue", 1.0);
                    	}
                    }
                    if(!currentTimeValue.isEmpty()&&!prev2TimeValue.isEmpty()){
                    	if(currentTimeValue.equals(prev2TimeValue)){
                    		valueSpecificFeatures.get(attribute).put("global_feature_specific_equal_to_prev2TimeValue", 1.0);
                    	}
                    	else{
                    		valueSpecificFeatures.get(attribute).put("global_feature_specific_Not_equal_to_prev2TimeValue", 1.0);
                    	}
                    }
                    if(!currentTimeValue.isEmpty()&&!prev3TimeValue.isEmpty()){
                    	if(currentTimeValue.equals(prev3TimeValue)){
                    		valueSpecificFeatures.get(attribute).put("global_feature_specific_equal_to_prev3TimeValue", 1.0);
                    	}
                    	else{
                    		valueSpecificFeatures.get(attribute).put("global_feature_specific_Not_equal_to_prev3TimeValue", 1.0);
                    	}
                    }
                    if(!currentTimeValue.isEmpty()&&!prev4TimeValue.isEmpty()){
                    	if(currentTimeValue.equals(prev4TimeValue)){
                    		valueSpecificFeatures.get(attribute).put("global_feature_specific_equal_to_prev4TimeValue", 1.0);
                    	}
                    	else{
                    		valueSpecificFeatures.get(attribute).put("global_feature_specific_Not_equal_to_prev4TimeValue", 1.0);
                    	}
                    }
                    if(!currentTimeValue.isEmpty()&&!prev5TimeValue.isEmpty()){
                    	if(currentTimeValue.equals(prev5TimeValue)){
                    		valueSpecificFeatures.get(attribute).put("global_feature_specific_equal_to_prev5TimeValue", 1.0);
                    	}
                    	else{
                    		valueSpecificFeatures.get(attribute).put("global_feature_specific_Not_equal_to_prev5TimeValue", 1.0);
                    	}
                    }
                } else {
                    valueSpecificFeatures.get(attribute).put("global_feature_specific_isNotInMR", 1.0);
                }
        		String prevAttr = "@@";
                if (mentionedAttr.size() - 1 >= 0) {
                    prevAttr = mentionedAttr.get(mentionedAttr.size() - 1).trim();
                }
        		//Is attr already mentioned right before
                if (prevAttr.equals(attribute)) {
                    valueSpecificFeatures.get(attribute).put("global_feature_specific_attrFollowingSameAttr", 1.0);
                } else {
                    valueSpecificFeatures.get(attribute).put("global_feature_specific_attrNotFollowingSameAttr", 1.0);
                }
                //Is attr already mentioned
                attrsAlreadyMentioned.stream().forEach((attr) -> {
                    if(attribute.equals(attr)){
                    	valueSpecificFeatures.get(attribute).put("global_feature_specific_attrAlreadyMentioned", 1.0);
                    }

                });
        	
                //Is attr to be mentioned (has value to express)
                boolean toBeMentioned = false;
                for(String attr :attrsToBeMentioned ){
                	if(attribute.equals(attr)){
                		toBeMentioned = true;
                        valueSpecificFeatures.get(attribute).put("global_feature_specific_attrToBeMentioned", 1.0);
                	}
                }
                if(!toBeMentioned){
                	valueSpecificFeatures.get(attribute).put("global_feature_specific_attrNotToBeMentioned", 1.0);
                }
        	}
        	HashSet<String> keys = new HashSet<>(valueSpecificFeatures.get(attribute).keySet());
            keys.forEach((feature1) -> {
                keys.stream().filter((feature2) -> (valueSpecificFeatures.get(attribute).get(feature1) == 1.0
                        && valueSpecificFeatures.get(attribute).get(feature2) == 1.0
                        && feature1.compareTo(feature2) < 0)).forEachOrdered((feature2) -> {
                    valueSpecificFeatures.get(attribute).put(feature1 + "&&" + feature2, 1.0);
                });
            });
            ArrayList<String> fullGramLM = new ArrayList<>();
            for(int i=0;i<mentionedAttr.size();i++){
            	fullGramLM.add(0,mentionedAttr.get(i));
            	
            }
            ArrayList<String> prev5attrGramLM = new ArrayList<>();
            int j=0;
            for (int i = mentionedAttr.size() - 1; (i >= 0 && j < 5); i--) {
                prev5attrGramLM.add(0, mentionedAttr.get(i));
                j++;
            }
            while (prev5attrGramLM.size() < 4) {
                prev5attrGramLM.add(0, "@@");
            }
            double afterLMScore = attrLMs.getProbability(fullGramLM);
            valueSpecificFeatures.get(attribute).put("global_feature_LMAttrFull_score", afterLMScore);
            afterLMScore = attrLMs.getProbability(prev5attrGramLM);
            valueSpecificFeatures.get(attribute).put("global_feature_LMAttr_score", afterLMScore);
        }
        return new Instance(generalFeatures, valueSpecificFeatures, costs);
	}
	
	public Instance createFieldInstance(String bestAction,ArrayList<String> previousGeneratedAttributes
			,ArrayList<String> previousGeneratedFields,ArrayList<String> nextGeneratedAttributes,HashSet<String> attrFieldValuesAlreadyMentioned
			,HashSet<String> attrFieldValuesThatFollow,HashMap<String,HashSet<String>> availableFieldPerAttr
			,WeatherMeaningRepresentation MR){
		TObjectDoubleHashMap<String> costs = new TObjectDoubleHashMap<>();
		if (!bestAction.isEmpty()) {
			if(bestAction.contains("=")){
				String attr = bestAction.split("=")[0];
				String field = bestAction;
				for(String action :availableFieldPerAttr.get(attr) ){
					if(action.equalsIgnoreCase(field)){
						costs.put(action, 0.0);
					}
					else{
						costs.put(action, 1.0);
					}
					
				}
				if(field.equals(WeatherAction.TOKEN_END)){
					costs.put(WeatherAction.TOKEN_END, 0.0);
				}else{
					costs.put(WeatherAction.TOKEN_END, 1.0);
				}
			}
			
		}
		return createFieldInstanceWithCosts(bestAction,costs,previousGeneratedAttributes,previousGeneratedFields
				,nextGeneratedAttributes,attrFieldValuesAlreadyMentioned,attrFieldValuesThatFollow
				,availableFieldPerAttr,MR);
		
	}
	public Instance createFieldInstanceWithCosts(String currentAttrField,TObjectDoubleHashMap<String> costs,ArrayList<String> previousGeneratedAttributes
			,ArrayList<String> previousGeneratedFields,ArrayList<String> nextGeneratedAttributes,HashSet<String> attrFieldValuesAlreadyMentioned
			,HashSet<String> attrFieldValuesThatFollow,HashMap<String,HashSet<String>> availableFieldPerAttr
			,WeatherMeaningRepresentation MR){
		
		TObjectDoubleHashMap<String> generalFeatures = new TObjectDoubleHashMap<>();
        HashMap<String, TObjectDoubleHashMap<String>> valueSpecificFeatures = new HashMap<>();
        String currentAttr = currentAttrField.split("=")[0];
        //String currentField  = currentAttrField;
        for(String action : availableFieldPerAttr.get(currentAttr)){
        	valueSpecificFeatures.put(action, new TObjectDoubleHashMap<String>());
        }
        ArrayList<String> generatedFields = new ArrayList<>();
        ArrayList<String> generatedFieldsInSameAttr = new ArrayList<>();
        String previousField = "";
        String previousAttr  = "";
        for(int w = 0;w<previousGeneratedFields.size();w++){
        	String field = previousGeneratedFields.get(w);
        	String attr ="";
        	if(!field.equals(WeatherAction.TOKEN_END)&&
        			!field.equals(WeatherAction.TOKEN_START)){
        	
        		//field = previousGeneratedFields.get(w).split("=")[0]
        			//+"="+previousGeneratedFields.get(w).split("=")[1];
        		attr = previousGeneratedFields.get(w).split("=")[0];
        		if(!attr.equals(previousAttr)){
        			previousAttr = attr;
        		}
        	}
        	else{
        		attr = previousAttr;
        	}
        	if(!previousField.equals(field)){
        		previousField = field;
        		generatedFields.add(field);
        	}
        	if(attr.equals(currentAttr)){
        		generatedFieldsInSameAttr.add(field);
        	}
        		
        		
        	
        	
        }
        
        //generatedFields in same attribute
        previousField = "@@";
        String previousFieldValue = "";
        if(generatedFieldsInSameAttr.size()-1>=0){
        	previousField = generatedFieldsInSameAttr.get(generatedFieldsInSameAttr.size()-1);
        	if(!previousField.equals(WeatherAction.TOKEN_END)
        			&&!previousField.equals(WeatherAction.TOKEN_START)
        			&&!previousField.equals("@@")){
        		String prevAttr = previousField.split("=")[0];
        		String prevField = previousField.split("=")[1];
        		if(MR.getAttrFieldValue().containsKey(prevAttr)){
        			String prevValue = MR.getAttrFieldValue().get(prevAttr).get(prevField);
        			previousFieldValue = previousField+"="+prevValue;
        		}
        	}
        }
        generalFeatures.put("feature_field_inSameAttr_"+previousField, 1.0);
        if(!previousFieldValue.isEmpty()){
        	generalFeatures.put("feature_fieldValue_inSameAttr_"+previousFieldValue, 1.0);
        }
        
        String previous2Field = "@@";
        String previous2FieldValue = "";
        if(generatedFieldsInSameAttr.size()-2>=0){
        	previous2Field = generatedFieldsInSameAttr.get(generatedFieldsInSameAttr.size()-2);
        	if(!previous2Field.equals(WeatherAction.TOKEN_END)
        			&&!previous2Field.equals(WeatherAction.TOKEN_START)
        			&&!previous2Field.equals("@@")){
        		String prev2Attr = previous2Field.split("=")[0];
        		String prev2field = previous2Field.split("=")[1];
        		if(MR.getAttrFieldValue().containsKey(prev2Attr)){
        			String prev2Value = MR.getAttrFieldValue().get(prev2Attr).get(prev2field);
        			previous2FieldValue = previous2Field+"="+prev2Value;
        		}
        		
        	}
        }
        generalFeatures.put("feature_prev2field_inSameAttr_"+previousField+"|"+previous2Field, 1.0);
        if(!previous2FieldValue.isEmpty()&&!previousFieldValue.isEmpty()){
        	generalFeatures.put("feature_prev2fieldValue_inSameAttr_"+previousFieldValue+"|"+previous2FieldValue, 1.0);
        }
        if(!previous2FieldValue.isEmpty()){
        	generalFeatures.put("feature_prev2SinglefieldValue_inSameAttr_"+previous2FieldValue, 1.0);
        }
        
        String previous3Field = "@@";
        String previous3FieldValue = "";
        if(generatedFieldsInSameAttr.size()-3>=0){
        	previous3Field = generatedFieldsInSameAttr.get(generatedFieldsInSameAttr.size()-3);
        	if(!previous3Field.equals(WeatherAction.TOKEN_END)
        			&&!previous3Field.equals(WeatherAction.TOKEN_START)
        			&&!previous3Field.equals("@@")){
        		String prev3Attr = previous3Field.split("=")[0];
        		String prev3field = previous3Field.split("=")[1];
        		if(MR.getAttrFieldValue().containsKey(prev3Attr)){
        			String prev3Value = MR.getAttrFieldValue().get(prev3Attr).get(prev3field);
        			previous3FieldValue = previous3Field+"="+prev3Value;
        		}
        		
        	}
        }       		
        generalFeatures.put("feature_prev3field_inSameAttr_"+previous3Field+"|"+previousField+"|"+previous2Field, 1.0);
        if(!previous3FieldValue.isEmpty()&&!previousFieldValue.isEmpty()&&!previous3FieldValue.isEmpty()){
        	generalFeatures.put("feature_prev3fieldValue_inSameAttr_"+previous3FieldValue+"|"+previousFieldValue+"|"+previous3FieldValue, 1.0);
        }
        if(!previous3FieldValue.isEmpty()){
        	generalFeatures.put("feature_prev3SinglefieldValue_inSameAttr_"+previous3FieldValue, 1.0);
        }
     // generated all fields
        previousField = "@@";
        previousFieldValue = "";
        if(generatedFields.size()-1>=0){
        	previousField = generatedFields.get(generatedFields.size()-1);
        	if(!previousField.equals(WeatherAction.TOKEN_END)
        			&&!previousField.equals(WeatherAction.TOKEN_START)
        			&&!previousField.equals("@@")){
        		String prevAttr = previousField.split("=")[0];
        		String prevField = previousField.split("=")[1];
        		if(MR.getAttrFieldValue().containsKey(prevAttr)){
        			String prevValue = MR.getAttrFieldValue().get(prevAttr).get(prevField);
        			previousFieldValue = previousField+"="+prevValue;
        		}
        		
        	}
        }
        generalFeatures.put("feature_field_prevInAll_"+previousField, 1.0);
        if(!previousFieldValue.isEmpty()){
        	generalFeatures.put("featuer_fieldValue_prevInAll"+previousFieldValue, 1.0);
        }
        previous2Field = "@@";
        previous2FieldValue = "";
        if(generatedFields.size()-2>=0){
        	previous2Field = generatedFields.get(generatedFields.size()-2);
        	if(!previous2Field.equals(WeatherAction.TOKEN_END)
        			&&!previous2Field.equals(WeatherAction.TOKEN_START)
        			&&!previous2Field.equals("@@")){
        		String prev2Attr = previous2Field.split("=")[0];
        		String prev2Field = previous2Field.split("=")[1];
        		if(MR.getAttrFieldValue().containsKey(prev2Attr)){
        			String prev2Value = MR.getAttrFieldValue().get(prev2Attr).get(prev2Field);
        			previous2FieldValue = previous2Field+"="+prev2Value;
        		}
        		
        	}
        }
        generalFeatures.put("feature_field_prev2InAll_"+previousField+"|"+previous2Field, 1.0);
        if(!previous2FieldValue.isEmpty()&&!previousFieldValue.isEmpty()){
        	generalFeatures.put("featuer_fieldValue_prev2InAll"+previousFieldValue+"|"+previous2FieldValue, 1.0);
        }
        if(!previous2FieldValue.isEmpty()){
        	generalFeatures.put("feature_fieldValue_prev2SingleInAll"+previous2FieldValue, 1.0);
        }
        previous3Field = "@@";
        previous3FieldValue = "";
        if(generatedFields.size()-3>=0){
        	previous3Field = generatedFields.get(generatedFields.size()-3);
        	if(!previous3Field.equals(WeatherAction.TOKEN_END)
        			&&!previous3Field.equals(WeatherAction.TOKEN_START)
        			&&!previous3Field.equals("@@")){
        		String prev3Attr = previous3Field.split("=")[0];
        		String prev3Field = previous3Field.split("=")[1];
        		if(MR.getAttrFieldValue().containsKey(prev3Attr)){
        			String prev3Value = MR.getAttrFieldValue().get(prev3Attr).get(prev3Field);
        			previous3FieldValue = previous3Field+"="+prev3Value;
        		}
        		
        	}
        }
        generalFeatures.put("feature_field_prev3InAll_"+previousField+"|"+previous2Field+"|"+previous3Field, 1.0);
        if(!previous3FieldValue.isEmpty()&&!previous2FieldValue.isEmpty()&&!previousFieldValue.isEmpty()){
        	generalFeatures.put("featuer_fieldValue_prev3InAll"+previousFieldValue+"|"+previous2FieldValue+"|"+previous3FieldValue, 1.0);
        }
        if(!previous3FieldValue.isEmpty()){
        	generalFeatures.put("feature_fieldValue_prev3SingleInAll"+previous3FieldValue, 1.0);
        }
        String previous4Field = "@@";
        String previous4FieldValue = "";
        if(generatedFields.size()-4>=0){
        	previous4Field = generatedFields.get(generatedFields.size()-4);
        	if(!previous4Field.equals(WeatherAction.TOKEN_END)
        			&&!previous4Field.equals(WeatherAction.TOKEN_START)
        			&&!previous4Field.equals("@@")){
        		String prev4Attr = previous4Field.split("=")[0];
        		String prev4Field = previous4Field.split("=")[1];
        		if(MR.getAttrFieldValue().containsKey(prev4Attr)){
        			String prev4Value = MR.getAttrFieldValue().get(prev4Attr).get(prev4Field);
        			previous4FieldValue = previous4Field+"="+prev4Value;
        		}
        		
        	}
        }
        generalFeatures.put("feature_field_prev4InAll_"+previousField+"|"+previous2Field+"|"+previous3Field+"|"+previous4Field, 1.0);
        if(!previous4FieldValue.isEmpty()&&!previous3FieldValue.isEmpty()&&!previous2FieldValue.isEmpty()&&!previousFieldValue.isEmpty()){
        	generalFeatures.put("featuer_fieldValue_prev4InAll"+previousFieldValue+"|"+previous2FieldValue+"|"+previous3FieldValue+"|"+previous4FieldValue, 1.0);
        }
        if(!previous4FieldValue.isEmpty()){
        	generalFeatures.put("feature_fieldValue_prev4SingleInAll"+previous4FieldValue, 1.0);
        }
        String previous5Field = "@@";
        String previous5FieldValue = "";
        if(generatedFields.size()-5>=0){
        	previous5Field = generatedFields.get(generatedFields.size()-5);
        	if(!previous5Field.equals(WeatherAction.TOKEN_END)
        			&&!previous5Field.equals(WeatherAction.TOKEN_START)
        			&&!previous5Field.equals("@@")){
        		String prev5Attr = previous5Field.split("=")[0];
        		String prev5Field = previous5Field.split("=")[1];
        		if(MR.getAttrFieldValue().containsKey(prev5Attr)){
        			String prev5Value = MR.getAttrFieldValue().get(prev5Attr).get(prev5Field);
        			previous5FieldValue = previous5Field+"="+prev5Value;
        		}
        		
        	}
        }
        generalFeatures.put("feature_field_prev5InAll_"+previousField+"|"+previous2Field+"|"+previous3Field+"|"+previous4Field+"|"+previous5Field, 1.0);
        if(!previous5FieldValue.isEmpty()&&!previous4FieldValue.isEmpty()&&!previous3FieldValue.isEmpty()&&!previous2FieldValue.isEmpty()&&!previousFieldValue.isEmpty()){
        	generalFeatures.put("featuer_fieldValue_prev5InAll"+previousFieldValue+"|"+previous2FieldValue+"|"+previous3FieldValue+"|"
        +previous4FieldValue+"|"+previous5FieldValue, 1.0);
        }
        if(!previous5FieldValue.isEmpty()){
        	generalFeatures.put("feature_fieldValue_prev5SingleInAll"+previous5FieldValue, 1.0);
        }
        // attribute field to be mentioned
        attrFieldValuesThatFollow.stream().forEach((attrField)->{
        	generalFeatures.put("feature_attrField_TBMentioned"+attrField, 1.0);
        });
        //attribute field already mentioned
        attrFieldValuesThatFollow.stream().forEach((attrField)->{
        	generalFeatures.put("feature_attrField_alreadyMentioned"+attrField, 1.0);
        });
        //generated attribute
        previousGeneratedAttributes.stream().forEach((attr)->{
        	generalFeatures.put("feature_attr_areadyMentioned"+attr,1.0 );
        });
        //next generated attribute
        previousGeneratedAttributes.stream().forEach((attr)->{
        	generalFeatures.put("feature_attr_TBMentioned"+attr,1.0);
        });
        for(String action:availableFieldPerAttr.get(currentAttr) ){
        	if(action.contains("=")){
        		String attr = action.split("=")[0];
        		if(MR.getAttrFieldValue().containsKey(attr)){
        			generalFeatures.put("feature_attrField_inMR"+action,1.0 );
        		}
        	}
        }
        for(String action: availableFieldPerAttr.get(currentAttr)){
        	if(action.equals(WeatherAction.TOKEN_END)){
        		if(attrFieldValuesAlreadyMentioned.isEmpty()){
        			valueSpecificFeatures.get(action).put("feature_valueSpecific_attrFieldValues_allMentioned", 1.0);
        		}
        		else{
        			valueSpecificFeatures.get(action).put("feature_valueSpecific_attrFieldValues_notAllMentioned", 1.0);
        		}
        	}else{
        		if(action.contains("=")){
        			String attr = action.split("=")[0];
        			if(MR.getAttrFieldValue().containsKey(attr)){
        				valueSpecificFeatures.get(action).put("feature_valueSpecific_inMR", 1.0);
        			}else{
        				valueSpecificFeatures.get(action).put("feature_valueSpecific_notInMR", 1.0);
        			}
        		}
        		if(action.equals(previousField)){
        			valueSpecificFeatures.get(action).put("feature_valueSpecific_equalPrevious1", 1.0);
        		}
        		if(action.equals(previous2Field)){
        			valueSpecificFeatures.get(action).put("feature_valueSpecific_equalPrevious2", 1.0);
        		}
        		if(attrFieldValuesThatFollow.contains(action)){
        			valueSpecificFeatures.get(action).put("feature_valueSpecific_toBeMentioned",1.0);
        		}
        		else{
        			valueSpecificFeatures.get(action).put("feature_valueSpecific_notToBeMentioned",1.0);
        		}
        	}
        	double score = fieldLMs.getProbability(generatedFields);
        	valueSpecificFeatures.get(action).put("feature_valueSpecific_fullGramLM", score);
        	score  = fieldLMs.getProbability(generatedFieldsInSameAttr);
        	valueSpecificFeatures.get(action).put("feature_valueSpecific_5GramLM", score);
        }
        
        
        
        
        
        return new Instance(generalFeatures, valueSpecificFeatures, costs);
	}
	
	
	
	
	
	public Object[] inferFeatureAndCostVectors() {

		ConcurrentHashMap<WeatherDatasetInstance,ArrayList<Instance>> attributeTrainingData = new ConcurrentHashMap<>();
		ConcurrentHashMap<WeatherDatasetInstance, HashMap<String,ArrayList<Instance>>> fieldTrainingData = new ConcurrentHashMap<>();
		ConcurrentHashMap<WeatherDatasetInstance, HashMap<String,HashMap<String,ArrayList<Instance>>>> wordTrainingData  = new ConcurrentHashMap<>();
		if(!availableWordAction.isEmpty()){
			trainingData.stream().forEach((di)->{
				attributeTrainingData.put(di, new ArrayList<>());
				fieldTrainingData.put(di, new HashMap<>());
				wordTrainingData.put(di, new HashMap<>());
				for(String attr: attributes){
					fieldTrainingData.get(di).put(attr, new ArrayList<>());
					wordTrainingData.get(di).put(attr, new HashMap<>());
					for(String field : attributeFields.get(attr)){
						wordTrainingData.get(di).get(attr).put(field, new ArrayList<>());
					}
					
				}
				
				
			});
		}
		
		Object[] results = new Object[2];
		return results;
	}
	
	
	
	
	public ArrayList<Instance> getAttributeTrainingData() {
		return attributeTrainingData;
	}
	public void setAttributeTrainingData(ArrayList<Instance> attributeTrainingData) {
		this.attributeTrainingData = attributeTrainingData;
	}
	public HashMap<String,ArrayList<Instance>> getFieldTrainingData() {
		return fieldTrainingData;
	}
	public void setFieldTrainingData(HashMap<String,ArrayList<Instance>> fieldTrainingData) {
		this.fieldTrainingData = fieldTrainingData;
	}
	public HashMap<String,HashMap<String,ArrayList<Instance>>> getWordTrainingData() {
		return wordTrainingData;
	}
	public void setWordTrainingData(HashMap<String,HashMap<String,ArrayList<Instance>>> wordTrainingData) {
		this.wordTrainingData = wordTrainingData;
	}



}
class InferWeatherVectorsThread extends Thread{
	WeatherDatasetInstance di;
	WeatherGov  wg;
	ConcurrentHashMap<WeatherDatasetInstance,ArrayList<Instance>> attributeTrainingData;
	ConcurrentHashMap<WeatherDatasetInstance, HashMap<String,ArrayList<Instance>>> fieldTrainingData;
	ConcurrentHashMap<WeatherDatasetInstance, HashMap<String,HashMap<String,ArrayList<Instance>>>> wordTrainingData;
	
	InferWeatherVectorsThread(WeatherDatasetInstance di,WeatherGov wg
			,ConcurrentHashMap<WeatherDatasetInstance,ArrayList<Instance>> attributeTrainingData
			,ConcurrentHashMap<WeatherDatasetInstance, HashMap<String,ArrayList<Instance>>> fieldTrainingData
			,ConcurrentHashMap<WeatherDatasetInstance, HashMap<String,HashMap<String,ArrayList<Instance>>>> wordTrainingData){
		this.di = di;
		this.wg = wg;
		this.attributeTrainingData = attributeTrainingData;
		this.fieldTrainingData = fieldTrainingData;
		this.wordTrainingData = wordTrainingData;
		
	}
	/**
     * This method goes through the ActionSequence one time-step at the time, and creates a feature and cost vector for each one.
     * Meanwhile it tracks the context information that the feature vector requires.
     */
	@Override
	public void run(){
		ArrayList<WeatherAction> refSequence = di.getDirectReferenceSequence();
		//Collections to track which attribute/field/value pairs have already be mentioned in the sequence and which are yet to be mentioned
		HashSet<String> attrFieldValuesToBeMentioned = new HashSet<>();
		HashSet<String> attrFieldValuesAlreadyMentioned = new HashSet<>();
		for(String attribute :di.getMR().getAttrFieldValue().keySet()){
			
				attrFieldValuesToBeMentioned.add(attribute);
			
		}
		// First we create the feature and cost vectors for the content actions
        ArrayList<String> attributeSequence = new ArrayList<>();
        String previousAttr = "";
        for(int w = 0;w<refSequence.size();w++){
        	if(!refSequence.get(w).getAttribute().equals(WeatherAction.TOKEN_PUNCT)
        			){
        		String currentAttr =refSequence.get(w).getAttribute(); 
        		if(refSequence.get(w).getAttribute().contains("=")){
        			currentAttr = refSequence.get(w).getAttribute().split("=")[0];
        		}
        		if(!currentAttr.equals(previousAttr)){
        			if(!currentAttr.isEmpty()){
            			attrFieldValuesToBeMentioned.remove(currentAttr);
            		}

            		Instance attributeTraininigVector = wg.createAttributeInstance(currentAttr, 
            				attributeSequence, attrFieldValuesAlreadyMentioned, attrFieldValuesToBeMentioned,
            				di.getMR(), wg.availableAttr);
            		if(attributeTraininigVector!=null){
            			attributeTrainingData.get(di).add(attributeTraininigVector);
            			
            		}
            		attributeSequence.add(currentAttr);
            		if(!currentAttr.isEmpty()){
            			attrFieldValuesToBeMentioned.remove(currentAttr);
            			attrFieldValuesAlreadyMentioned.add(currentAttr);
            		}
        		}
        		
        		
        		
        		
        		
        		
        		
        		
        	}	
        	
        }
        //Reset the tracking collections
        attrFieldValuesToBeMentioned = new HashSet<>();
		attrFieldValuesAlreadyMentioned = new HashSet<>();
		for(String attribute :di.getMR().getAttrFieldValue().keySet()){
			for(String field :di.getMR().getAttrFieldValue().get(attribute).keySet()){
				
				attrFieldValuesToBeMentioned.add(attribute+"="+field);
			}
		}
		
		// Then we create the feature and cost vectors for the field actions
        // Each field action corresponds to a attribute action, so we need to keep track of which attribute action we are "generating" from at each timestep
		
		//the sequence of already generated attrFieldValue pairs
		ArrayList<String> attrs = new ArrayList<>();
        
        
        // current attribute
        String attribute = "";
        String fieldTBMentioned = "";
        
        // Time-step counter
        int a = -1;
        // This tracks the subphrase consisting of the fields generated for the current attribute action
        ArrayList<String> fieldSequence = new ArrayList<>();
        // For every step of the sequence
        for (int w = 0; w < refSequence.size(); w++) {
        	if(!refSequence.get(w).getAttribute().equals(WeatherAction.TOKEN_PUNCT)){
        		// If this action does not belong to the current attribute, we need to update the trackers and switch to the new attribute action
        		// if current attribute is changed, we need to update
        		String currentAttr = refSequence.get(w).getAttribute();
        		if(refSequence.get(w).getAttribute().contains("=")){
        			currentAttr = refSequence.get(w).getAttribute().split("=")[0];
        		}
        		if(!currentAttr.equals(attribute)){
        			a++;
        			currentAttr = attribute;
        			
        			attrs.add(attribute);
        			
        			
        		}
        		String currentField = refSequence.get(w).getField();
        		if(refSequence.get(w).getField().contains("=")){
        			currentField = refSequence.get(w).getField().split("=")[0]+"="+refSequence.get(w).getField().split("=")[1];
        		}
        		if(!currentField.equals(fieldTBMentioned)){
        			fieldTBMentioned = currentField;
        			// The subsequence of attrFieldValue pairs we have generated for so far
                    ArrayList<String> predictedAttributesForInstance = new ArrayList<>();
                    for (int i = 0; i < attrs.size(); i++) {
                        predictedAttributesForInstance.add(attrs.get(i));
                    }
                    ArrayList<String> nextAttributesForInstance = new ArrayList<>();
                    // Create the feature and cost vector
                    for(int k = a+1;k<attributeSequence.size();k++){
                    	nextAttributesForInstance.add(attributeSequence.get(k));
                    }
                    Instance fieldTrainingVector = wg.createFieldInstance(fieldTBMentioned, predictedAttributesForInstance
                    		, fieldSequence, nextAttributesForInstance, attrFieldValuesAlreadyMentioned
                    		, attrFieldValuesToBeMentioned, wg.availableFieldPerAttr, di.getMR());
                    if(fieldTrainingVector!=null){
                    	fieldTrainingData.get(di).get(currentAttr).add(fieldTrainingVector);
                    }
                    if(!fieldTBMentioned.isEmpty()){
                    	attrFieldValuesAlreadyMentioned.add(fieldTBMentioned);
                    	attrFieldValuesToBeMentioned.remove(fieldTBMentioned);
                    }
                    fieldSequence.add(currentField);
                    
        		}
        		
        		
            		
            	
            	
        	}
        	
        	
        }
        
        
		
		
		
		
		
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
}
