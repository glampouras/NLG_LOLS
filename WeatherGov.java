package WeatherGov ;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.regex.Pattern;

import structuredPredictionNLG.Action;





public class WeatherGov   {
	HashMap<String,HashMap<String,HashSet<String>>> attributeFieldValuePairs = new HashMap<>();
	Boolean useAlignmentData = true;
	HashSet<String> attributes = new HashSet<>();
	HashMap<String,HashSet<String>> attributeFields = new HashMap<>();
	HashMap<String,HashSet<String>> fieldValues = new HashMap<>();
	int maxWordSequenceLength = 0;
	public static void main(String[] args){
		// TODO Auto-generated method stub
		WeatherGov w = new WeatherGov();
		ArrayList<String> fileNames = w.readInDataName();
		w.readInFile(fileNames);
		
		
		
		
		
	}
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
	
	
	
	
	
	public void readInFile(ArrayList<String> fileNames){
		HashMap<String,ArrayList<String>>  docToEvents = new HashMap<>();
		HashMap<String,ArrayList<String>>  docToAlign = new HashMap<>();
		HashMap<String,ArrayList<String>>  docToText = new HashMap<>();
		BufferedReader br = null;
		for(String name : fileNames){
			if(name.contains(".events")){
				try {
					
					br = new BufferedReader(new FileReader(name));
				} catch (Exception e1) {
					// TODO Auto-generated catch block
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
					// TODO Auto-generated catch block
					e.printStackTrace();
					System.out.println("can not find file");
				}finally{
					try {
						br.close();
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}else if(name.contains(".align")){
				try {
					br = new BufferedReader(new FileReader(name));
					//er = new EasyReader(name);
				} catch (Exception e1) {
					// TODO Auto-generated catch block
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
					// TODO Auto-generated catch block
					e.printStackTrace();
					System.out.println("can not find file");
				}finally{
					try {
						br.close();
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}else if(name.contains(".text")){
				try {
					br = new BufferedReader(new FileReader(name));
					//er = new EasyReader(name);
				} catch (Exception e1) {
					// TODO Auto-generated catch block
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
					// TODO Auto-generated catch block
					e.printStackTrace();
					System.out.println("can not find file");
				}finally{
					try {
						br.close();
					} catch (IOException e) {
						// TODO Auto-generated catch block
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
		
		/*for every instance , generate Meaning Representation and reference (dataInstance)
		 * 
		 * */
		
		for(String docName : docToEvents.keySet()){
			
			String attributeId;
			ArrayList<String> ref = new ArrayList<>() ;
			String MRstr = " ";
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
				
				if(!fields[fields.length-1].equals("@mode:--")){
					attributeId = fields[0].split(":")[1];
					attributes.add(attributeId);
					if(!attributeFields.containsKey(attributeId)){
						attributeFields.put(attributeId, new HashSet<String>());
					}
					for(int i = 1;  i<fields.length;i++){
						/*
						 * get out the data: label: and Night , no use. 
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
								String value = " ";
								String field = " ";
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
				
			}
			
			HashMap<String,HashSet<String>> textAttrIdAlignment = new  HashMap<>();//for each instance
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
			//  for each instance build MR
			WeatherMeaningRepresentation MR = new WeatherMeaningRepresentation(attrFieldValue,MRstr);
			
			//start build DatasetInstance
			ArrayList<String> observedAttrFieldValueSequence = new ArrayList<>();
            ArrayList<String> observedWordSequence = new ArrayList<>();
            String refStr = String.join("", ref);
            String[] words = refStr.replaceAll("([.,])", " @punc").split("\\s+");
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
                if (word.trim().matches("@punc")) {
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
                directReferenceSequence.add(new WeatherAction(observedWordSequence.get(r),"",wordToAttrFieldValueAlignment.get(r)));
            }  
            for(int i = 0; i<directReferenceSequence.size();i++){
            	WeatherDatasetInstance DI = new WeatherDatasetInstance(MR,directReferenceSequence,"");
            }
            
				
					
					
				
				
			
			
			
			
			
		}//end for each instance
		
	}
	public String postProcessRef(WeatherMeaningRepresentation mr, ArrayList<WeatherAction> directReferenceSequence){
		String cleanedWords = "";
		
		
		return "";
	}

}
