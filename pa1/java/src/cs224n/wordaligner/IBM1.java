package cs224n.wordaligner;  

import cs224n.util.*;
import java.util.List;

/**
 * IBM1 models the problem
 * 
 * @author Francois Chaubard
 */

public class IBM1 implements WordAligner {

	// Count of sentence pairs that contain source and target words
	private CounterMap<String,String> source_target_prob; // p(f|e)
	// Count of source words appearing in the training set
	//private Counter<String> source_count;

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
				double ai = source_target_prob.getCount(source, target) ;

				if (currentMax < ai){
					currentMax = ai;
					maxSourceIdx = srcIndex;
				}
			}
			
			alignment.addPredictedAlignment(targetIdx, maxSourceIdx);
		}
		return alignment;
	}

	public void setAllInCounterMap(List<SentencePair> trainingPairs, CounterMap<String,String> counterMap, double initValue){
		for(SentencePair pair : trainingPairs){
			List<String> targetWords = pair.getTargetWords();
			List<String> sourceWords = pair.getSourceWords();
			//Add a Null word to the source list
			sourceWords.add(NULL_WORD);
			for(String source : sourceWords){
				for(String target : targetWords){
					counterMap.setCount(source, target, initValue);
				}
			}
		}
	}
	
	public void train(List<SentencePair> trainingPairs) {
		//Initalize counters
		source_target_prob= new CounterMap<String,String>(); // p(f|e)
		CounterMap<String,String> source_target_count = new CounterMap<String,String>(); // c(f|e)
		boolean converged = false;
		
		// initialize the probability to uniform
		setAllInCounterMap(trainingPairs,source_target_prob,1.0);
		source_target_prob = Counters.conditionalNormalize(source_target_prob);
		
		double posterior_sum = 0.0;
		int count=0;
		while(!converged){
			count++;

			source_target_count = new CounterMap<String,String>(); 
			
			//For each sentence pair increment the counters
			for(SentencePair pair : trainingPairs){
				List<String> targetWords = pair.getTargetWords();
				List<String> sourceWords = pair.getSourceWords();
				for(String target : targetWords){
					posterior_sum = 0.0;
					for(String source : sourceWords){
						posterior_sum+=source_target_prob.getCount(source, target);
					}
							
					for(String source : sourceWords){
						source_target_count.incrementCount(source, target,  (source_target_prob.getCount(source, target)/posterior_sum));
					}
				}
			}
			
			// normalize the probabilities
			source_target_count = Counters.conditionalNormalize(source_target_count);
			
			// check if converged
			double error =0.0;
			for(SentencePair pair : trainingPairs){
				List<String> targetWords = pair.getTargetWords();
				List<String> sourceWords = pair.getSourceWords();
				for(String source : sourceWords){
					for(String target : targetWords){
						error += Math.pow(source_target_count.getCount(source, target) - source_target_prob.getCount(source, target) ,2);
					}
				}
			}
			if (error < 0.5 | count > 100){
				converged=true;
			}
			
			source_target_prob = source_target_count;
				
			System.out.printf("iteration number %d  error %f \n", count, error );
			
			
		}
	}
}

