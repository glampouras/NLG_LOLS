package WeatherGov;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;


public class WeatherAction implements Serializable {
	private static final long serialVersionUID = 1L;
	final public static String TOKEN_START = "@start@";
	final public static String TOKEN_END = "@end@";
	final public static String TOKEN_PUNCT = "@punct@";
	private String word;
	private String field;
    private String attribute;
    HashSet<String> attrFieldValuesBeforeThisTimestep_InContentSequence;
    HashSet<String> attrFieldValuesAfterThisTimestep_InContentSequence;
    ArrayList<String> attrFieldValuesBeforeThisTimestep_InWordSequence;
    ArrayList<String> attrFieldValuesAfterThisTimestep_InWordSequence;
    boolean isValueMentionedAtThisTimestep;
    public WeatherAction(String word, String field,String attribute) {
        this.setWord(word);
        this.setAttribute(attribute);
        this.setField(field);
    }
        public WeatherAction(WeatherAction a) {
        this.setWord(a.getWord());
        this.setAttribute(a.getAttribute());
        
        if (a.getattrFieldValuesBeforeThisTimestep_InContentSequence() != null) {
            this.attrFieldValuesBeforeThisTimestep_InContentSequence = new HashSet<>(a.getattrFieldValuesBeforeThisTimestep_InContentSequence());
        }
        if (a.getattrFieldValuesAfterThisTimestep_InContentSequence() != null) {
            this.attrFieldValuesAfterThisTimestep_InContentSequence = new HashSet<>(a.getattrFieldValuesAfterThisTimestep_InContentSequence());
        }
        if (a.getattrFieldValuesBeforeThisTimestep_InWordSequence() != null) {
            this.attrFieldValuesBeforeThisTimestep_InWordSequence = new ArrayList<>(a.getattrFieldValuesBeforeThisTimestep_InWordSequence());
        }
        if (a.getattrFieldValuesAfterThisTimestep_InWordSequence() != null) {
            this.attrFieldValuesAfterThisTimestep_InWordSequence = new ArrayList<>(a.getattrFieldValuesAfterThisTimestep_InWordSequence());
        }
        this.isValueMentionedAtThisTimestep = a.isValueMentionedAtThisTimestep;
    }
    public String getAction() {
        return word.toLowerCase().trim();
    }
	public String getWord() {
		return word;
	}
	public void setWord(String word) {
		this.word = word;
	}
	public String getField() {
		return field;
	}
	public void setField(String field) {
		this.field = field;
	}
	public String getAttribute() {
		return attribute;
	}
	public void setAttribute(String attribute) {
		this.attribute = attribute;
	}
	public HashSet<String> getattrFieldValuesBeforeThisTimestep_InContentSequence() {
        return attrFieldValuesBeforeThisTimestep_InContentSequence;
    }
	public HashSet<String> getattrFieldValuesAfterThisTimestep_InContentSequence() {
        return attrFieldValuesAfterThisTimestep_InContentSequence;
    }
	public ArrayList<String> getattrFieldValuesBeforeThisTimestep_InWordSequence() {
        return attrFieldValuesBeforeThisTimestep_InWordSequence;
    }
	public ArrayList<String> getattrFieldValuesAfterThisTimestep_InWordSequence() {
        return attrFieldValuesAfterThisTimestep_InWordSequence;
    }
	public boolean isValueMentionedAtThisTimestep() {
        return isValueMentionedAtThisTimestep;
    }
	
}
