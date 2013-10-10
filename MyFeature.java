package edu.stanford.nlp.mt.decoder.feat;

import java.util.*;
import edu.stanford.nlp.mt.base.*;



public class MyFeature implements RuleFeaturizer<IString,String> {
    private static final String FEATURE_NAME = "MYF";
    @Override
	public void initialize() {}
    @Override
	public List<FeatureValue<String>> ruleFeaturize(Featurizable<IString,String> f) {
	List<FeatureValue<String>> features =
	    new LinkedList<FeatureValue<String>>();

	int delta_size = f.sourcePhrase.size() - f.targetPhrase.size();
	if (f.sourcePhrase.size()>2 && (delta_size ==0 || delta_size == 1)){
	    features.add(new FeatureValue<String>(FEATURE_NAME,1.0));
	}
	
	return features;
    }
}
