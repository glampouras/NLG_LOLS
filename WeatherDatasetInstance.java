package WeatherGov;

import java.io.Serializable;
import java.util.ArrayList;

import java.util.HashSet;

import structuredPredictionNLG.DatasetInstance;

public class WeatherDatasetInstance implements Serializable, Comparable<DatasetInstance>{
	private static final long serialVersionUID = 1L;
	private WeatherMeaningRepresentation MR ;
	private ArrayList<WeatherAction> directReferenceSequence ;
	private ArrayList<WeatherAction> directAttrSequence ;
	private ArrayList<WeatherAction> directFieldSequence ;
	private String directReference ;
	private HashSet<String> evaluationReferences ;
	
	public WeatherDatasetInstance(WeatherMeaningRepresentation MR, ArrayList<WeatherAction> directReferenceSequence, String directReference){
		this.MR=MR;
		this.directReferenceSequence = directReferenceSequence;
		this.directReference=directReference;
		this.evaluationReferences = new HashSet<>();
		this.evaluationReferences.add(directReference);
	}
	public WeatherDatasetInstance(WeatherDatasetInstance di){
		this.MR =  di.getMR();
		this.directReferenceSequence = di.getDirectReferenceSequence();
		this.directReference = di.getDirectReference();
		this.evaluationReferences.addAll(di.getEvaluationReferences());
	}

	public HashSet<String> getEvaluationReferences() {
		
		return evaluationReferences;
	}
	@Override
	public int compareTo(DatasetInstance o) {
		// TODO Auto-generated method stub
		return 0;
	}

	public WeatherMeaningRepresentation getMR() {
		return MR;
	}

	

	public ArrayList<WeatherAction> getDirectReferenceSequence() {
		return directReferenceSequence;
	}
	
	public void setDirectReferenceSequence(ArrayList<WeatherAction> directReferenceSequence){
		this.directReferenceSequence = directReferenceSequence;
		/*
		 * get the attribute reference sequence and the field reference sequence
		 * 
		 * */
	}
	
	public ArrayList<WeatherAction> getDirectAttrReferenceSequence(){
		return this.directAttrSequence;
	}
	public ArrayList<WeatherAction> getDirectFieldReferenceSequence(){
		return this.directFieldSequence;
	}

	public String getDirectReference() {
		return directReference;
	}
}
	
