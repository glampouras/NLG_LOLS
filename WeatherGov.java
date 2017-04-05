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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import similarity_measures.Levenshtein;









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
			
			
			
			String attributeId;
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
					attributeId = fields[0].split(":")[1];
					attributes.add(attributeId);
					if(!attributeFields.containsKey(attributeId)){
						attributeFields.put(attributeId, new HashSet<String>());
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
									if(!attributeFieldValuePairs.containsKey(attributeId)){
										attributeFieldValuePairs.put(attributeId, new HashMap<>());
									} 
									if(!attributeFieldValuePairs.get(attributeId).containsKey(field)){
										attributeFieldValuePairs.get(attributeId).put(field, new HashSet<>());
									}
										
									attributeFieldValuePairs.get(attributeId).get(field).add(value);
									
									//populate attrFieldValue
									//for every MR
									if(!attrFieldValue.containsKey(attributeId)){
										
										attrFieldValue.put(attributeId, new HashMap<>());
									}
									if(!attrFieldValue.get(attributeId).containsKey(field)){
										attrFieldValue.get(attributeId).put(field,value);
									}
									attributeFields.get(attributeId).add(field);
									
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
			
			
			
			
			//  for each instance build MR
			WeatherMeaningRepresentation MR = new WeatherMeaningRepresentation(attrFieldValue,MRstr);
			
			//start build DatasetInstance
			ArrayList<String> observedAttrFieldValueSequence = new ArrayList<>();
            ArrayList<String> observedWordSequence = new ArrayList<>();
            String refStr = String.join("", ref);
            String[] words = refStr.replaceAll("([.,])", WeatherAction.TOKEN_PUNCT).split("\\s+");
            for(String w : words){
            	//if((!w.isEmpty())&&(observedWordSequence.isEmpty()||w.trim().equals(observedWordSequence.get(observedWordSequence.size()-1)if((!w.isEmpty())&&(observedWordSequence.isEmpty()||w.trim().equals(observedWordS))){
            		observedWordSequence.add(w.trim().toLowerCase());
            	//}
            }
            //System.out.println(observedWordSequence);
            if(observedWordSequence.size()>maxWordSequenceLength){
            	maxWordSequenceLength = observedWordSequence.size();
            }
            ArrayList<String> wordToAttrFieldValueAlignment = new ArrayList<>();
            observedWordSequence.forEach((word) -> {
                if (word.trim().matches(WeatherAction.TOKEN_PUNCT) ){
                    wordToAttrFieldValueAlignment.add(WeatherAction.TOKEN_PUNCT);
                } else {
                    wordToAttrFieldValueAlignment.add("[]");
                }
            });
            if(wordToAttrFieldValueAlignment.size()!=observedWordSequence.size()){
            	System.out.println("length not equal");
            }
            ArrayList<WeatherAction> directReferenceSequence = new ArrayList<>();
            for (int r = 0; r < observedWordSequence.size(); r++) {
                directReferenceSequence.add(new WeatherAction(observedWordSequence.get(r),wordToAttrFieldValueAlignment.get(r),""));
            } 
            /*
             * build DI
             * */
            
            WeatherDatasetInstance DI = new WeatherDatasetInstance(MR,directReferenceSequence,"");
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
            						||Pattern.matches("([0-9]+)-([0-9]+)",MR.getAttrFieldValue().get(attr).get(field) ))){
            			
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
					&&nlWord.equals(new WeatherAction("","",WeatherAction.TOKEN_END))){
				cleanedWords += " " + nlWord.getWord();
				
			}
			
			
			
		}
		cleanedWords = cleanedWords.trim()+ ".";
		
		
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
                
				
			});
			
		});
		
		
		
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
            		
            		if(a.getField().equals(WeatherAction.TOKEN_PUNCT)){
            			//System.out.println(a.getField());
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
                						||Pattern.matches("([0-9]+)-([0-9]+)",value ))&&valueAlignments.containsKey(value)){
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
                                    randomRealization.get(index).setAttribute(attrFieldValue.split("=")[0].toLowerCase().trim());
                                    randomRealization.get(index).setField(attrFieldValue.split("=")[1].toLowerCase().trim());
                                    
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
                }/*
                for(int i=0;i<endAttrRealization.size();i++){
                	System.out.print(endAttrRealization.get(i).getAttribute()+":"+endAttrRealization.get(i).getWord()+" ");
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
		
		//System.out.println(maxAttributeSequenceLength+" "+maxFieldSequenceLength);
		System.out.println("finish!");
	}
	

}
