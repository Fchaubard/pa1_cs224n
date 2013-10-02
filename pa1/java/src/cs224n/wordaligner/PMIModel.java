package cs224n.wordaligner;  

import cs224n.util.*;
import java.util.List;

/**
 * Pointwise mutual information (PMI) model. Aligns words based on the
 * frequency with which they occur in sentence pairs 
 * 
 * @author Colin Mayer
 */

public class PMIModel implements WordAligner {

	// Count of sentence pairs that contain source and target words
	private CounterMap<String,String> source_target_count;
	// Count of source words appearing in the training set
	private Counter<String> source_count;

	public Alignment align(SentencePair sentencePair) {
		Alignment alignment = new Alignment();
		
		int numSourceWords = sentencePair.getSourceWords().size();
		int numTargetWords = sentencePair.getTargetWords().size();

		for (int targetIdx = 0; targetIdx < numTargetWords; targetIdx++) {
			String target = sentencePair.getTargetWords().get(targetIdx);
			//Find source word with maximum alignment likelihood
			double currentMax = 0;
			int maxSourceIdx = 0;
			for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
				String source = sentencePair.getSourceWords().get(srcIndex);
				double ai = source_target_count.getCount(source, target) / source_count.getCount(source);

				if (currentMax < ai){
					currentMax = ai;
					maxSourceIdx = srcIndex;
				}
			}
			// Add the alignment
			alignment.addPredictedAlignment(targetIdx, maxSourceIdx);
		}
		return alignment;
	}

	public void train(List<SentencePair> trainingPairs) {
		//Initalize counters
		source_target_count = new CounterMap<String,String>();
		source_count = new Counter<String>();

		//For each sentence pair increment the counters
		for(SentencePair pair : trainingPairs){
			List<String> targetWords = pair.getTargetWords();
			List<String> sourceWords = pair.getSourceWords();
			//Add a Null word to the source list
			sourceWords.add(NULL_WORD);
			for(String source : sourceWords){
				source_count.incrementCount(source, 1.0);
				for(String target : targetWords){
					//Only count each word pair once
					if (sourceWords.indexOf(source) == sourceWords.lastIndexOf(source) &&
							targetWords.indexOf(target) == targetWords.lastIndexOf(target)){
							source_target_count.incrementCount(source, target, 1.0);
					}
				}
			}
		}
	}
}
