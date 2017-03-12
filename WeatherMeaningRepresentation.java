package WeatherGov;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Objects;

public class WeatherMeaningRepresentation implements Serializable {
	private static final long serialVersionUID = 1L;
	private  String MRstr;
    private final HashMap<String, HashMap<String,String>> attrFieldValue;
    
    public WeatherMeaningRepresentation(HashMap<String, HashMap<String,String>> attrFieldValue, String MRstr) {
        this.setMRstr(MRstr);
        this.attrFieldValue = attrFieldValue;
    }

	public String getMRstr() {
		return MRstr;
	}

	public void setMRstr(String mRstr) {
		MRstr = mRstr;
	}

	public HashMap<String, HashMap<String,String>> getAttrFieldValue() {
		return attrFieldValue;
	}
	@Override
    public int hashCode() {
        int hash = 7;
        hash = 61 * hash + Objects.hashCode(this.MRstr);
        hash = 61 * hash + Objects.hashCode(this.attrFieldValue);
        return hash;
    }
	private String abstractMR = "";
	public String getAbstractMR() {
        if (abstractMR.isEmpty()) {
         
            ArrayList<String> attrs = new ArrayList<>(attrFieldValue.keySet());
            Collections.sort(attrs);
            
            attrs.stream().map((attr) -> {
                abstractMR += attr + "={";
                return attr;
            }).forEach((attr) ->{
            	ArrayList<String> fields = new ArrayList<>(attrFieldValue.get(attr).keySet());
            	Collections.sort(fields);
            	fields.stream().map((field)->{
            		abstractMR += "["+field + ":";
            		return field;
            	}).forEach((field)->{
            		String value = attrFieldValue.get(attr).get(field);
            		abstractMR += value + "]";
            		
            	});
            	abstractMR += "}";	
            });
        }
        return abstractMR;
    }
	@Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final WeatherMeaningRepresentation other = (WeatherMeaningRepresentation) obj;
        if (!Objects.equals(this.MRstr, other.MRstr)) {
            return false;
        }
        return this.attrFieldValue.equals(other.attrFieldValue);
    }
	
}
