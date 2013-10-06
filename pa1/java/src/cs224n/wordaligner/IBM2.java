package cs224n.wordaligner;  

import cs224n.util.*;
import java.util.List;

/**
 * IBM2 models the problem 
 * 
 * @author Francois Chaubard
 */

public class IBM2 implements WordAligner {

	// Count of sentence pairs that contain source and target words
	private CounterMap<String,String> target_source_prob; // p(f|e)
	private CounterMap<String,String> target_source_tML; // t(fi|ej)
	private CounterMap<Pair<Integer,Integer>,Pair<Integer,Integer>> l_m_i_j_qML; // q(j|i,m,l) 
	
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
				double ai = target_source_prob.getCount(target, source) ;

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
					counterMap.setCount(target,  source, initValue);
				}
			}
		}
	}
	
	public void train(List<SentencePair> trainingPairs) {
		//Initalize counters
		target_source_prob= new CounterMap<String,String>(); // p(f|e)
		CounterMap<String,String> target_source_count = new CounterMap<String,String>(); // c(f|e)
		boolean converged = false;
		
		// initialize the probability to uniform
		setAllInCounterMap(trainingPairs,target_source_prob,1.0);
		target_source_prob = Counters.conditionalNormalize(target_source_prob);
		
		double posterior_sum = 0.0;
		int count=0;
		while(!converged){
			count++;

			target_source_count = new CounterMap<String,String>(); 
			
			//For each sentence pair increment the counters
			for(SentencePair pair : trainingPairs){
				List<String> targetWords = pair.getTargetWords();
				List<String> sourceWords = pair.getSourceWords();
				for(String source : sourceWords){
					posterior_sum = 0.0;
					for(String target : targetWords){
						posterior_sum+=target_source_prob.getCount(target, source);
					}
							
					for(String target : targetWords){
						target_source_count.incrementCount(target, source,  (target_source_prob.getCount(target, source)/posterior_sum));
					}
				}
			}
			
			// normalize the probabilities
			target_source_count = Counters.conditionalNormalize(target_source_count);
			
			// check if converged
			double error =0.0;
			for(SentencePair pair : trainingPairs){
				List<String> targetWords = pair.getTargetWords();
				List<String> sourceWords = pair.getSourceWords();
				for(String source : sourceWords){
					for(String target : targetWords){
						error += Math.pow(target_source_count.getCount(target, source) - target_source_prob.getCount(target, source) ,2);
					}
				}
			}
			if (error < 0.5 | count > 100){
				converged=true;
			}
			
			target_source_prob = target_source_count;
				
			System.out.printf("iteration number %d  error %f \n", count, error );
			
			
		}// while for Model 1
		
		System.out.printf("Finished with Model 1 \n" );
		
		
		
		
		
		// setup for Model 2
		l_m_i_j_qML = new CounterMap<Pair<Integer,Integer>,Pair<Integer,Integer>>(); // c(f|e)
		target_source_tML = new CounterMap<String,String>();
		
		// init t to probs from model 1
		target_source_tML = target_source_prob;
		
		// init l_m_i_j_count with Model 1 results
		CounterMap<Pair<Integer,Integer>,Pair<Integer,Integer>> l_m_i_j_count = new CounterMap<Pair<Integer,Integer>,Pair<Integer,Integer>>(); // c(f|e)
		System.out.printf("Starting Model 2 \n" );
		for(SentencePair pair : trainingPairs){
			
			int numSourceWords = pair.getSourceWords().size();
			int numTargetWords = pair.getTargetWords().size();
				
			for (int targetIdx = 0; targetIdx < numTargetWords; targetIdx++) {
				String target = pair.getTargetWords().get(targetIdx);
				//Find source word with maximum alignment likelihood
				double currentMax = 0;
				int maxSourceIdx = 0;
				for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
					String source = pair.getSourceWords().get(srcIndex);
					double ai = target_source_prob.getCount(target, source) ;

					if (currentMax < ai){
						currentMax = ai;
						maxSourceIdx = srcIndex;
					}
				}
				
				l_m_i_j_count.incrementCount(getPairOfInts(numTargetWords,numSourceWords),    
											 getPairOfInts(maxSourceIdx, targetIdx ),1);
			}
		}// sentences
		
		// now init qML
		for(Pair<Integer,Integer> key1: l_m_i_j_count.keySet()){
			
			double counter_ilm = 0;
			
			for (Pair<Integer,Integer> key2: l_m_i_j_count.getCounter(key1).keySet()){
				counter_ilm += l_m_i_j_count.getCount(key1, key2);
			}
			for (Pair<Integer,Integer> key2: l_m_i_j_count.getCounter(key1).keySet()){
				l_m_i_j_qML.setCount(key1, key2, ( l_m_i_j_count.getCount(key1, key2) /counter_ilm) );
			}
			
		}// sentences
		System.out.printf("Finished init for Model 2 \n" );
		
		// end of init
		
		
		
		
		
		
		
		// begin training Model 2
		for (int c=0; c<2000; c++ ){ // when to break out of loop TODO make this better
			System.out.printf("%d ",c );
			for(SentencePair pair : trainingPairs){
				
				int numSourceWords = pair.getSourceWords().size();
				int numTargetWords = pair.getTargetWords().size();
				
				for (int targetIdx = 0; targetIdx < numTargetWords; targetIdx++) {
					String target = pair.getTargetWords().get(targetIdx);
					//Find source word with maximum alignment likelihood
					double currentMax = 0;
					int maxSourceIdx = 0;
					for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
						String source = pair.getSourceWords().get(srcIndex);
						double ai = l_m_i_j_qML.getCount(getPairOfInts(numTargetWords,numSourceWords),getPairOfInts(srcIndex,targetIdx)) * target_source_tML.getCount(target, source) ;

						if (currentMax < ai){
							currentMax = ai;
							maxSourceIdx = srcIndex;
						}
					}
					
					// perform delta = ... 
					
					double count_ijlm =  l_m_i_j_count.getCount(getPairOfInts(numTargetWords, numSourceWords ),    
																getPairOfInts(maxSourceIdx,targetIdx));
					double delta_denominator_sum = 0.0;
					for (int targ = 0; targ < numTargetWords; targ++) {
						delta_denominator_sum +=l_m_i_j_count.getCount(getPairOfInts(numTargetWords,numSourceWords),    
																		getPairOfInts(maxSourceIdx,targ)) *
												target_source_tML.getCount( pair.getTargetWords().get(targ), pair.getSourceWords().get(maxSourceIdx) );
					}
					double newCount_ijlm = count_ijlm *  target_source_tML.getCount( target, pair.getSourceWords().get(maxSourceIdx)  ) / delta_denominator_sum;
					
					l_m_i_j_count.incrementCount(getPairOfInts(numTargetWords, numSourceWords ),    
							                     getPairOfInts(maxSourceIdx, targetIdx ), newCount_ijlm);
					
					target_source_count.incrementCount(pair.getTargetWords().get(targetIdx), pair.getSourceWords().get(maxSourceIdx),newCount_ijlm);
	
				}
				
			}// sentences
			
			//normalize count to become probs
			target_source_count = Counters.conditionalNormalize(target_source_count);
			
			
			// do we need to represent c(jilm) and c(ilm) seperately as well do we need to rep c(f,e) and c(e) separately?
			double error =0.0;
			for(SentencePair pair : trainingPairs){
				List<String> targetWords = pair.getTargetWords();
				List<String> sourceWords = pair.getSourceWords();
				for(String source : sourceWords){
					for(String target : targetWords){
						error += Math.pow(target_source_count.getCount(target, source) - target_source_count.getCount(target, source) ,2);
					}
				}
			}
			target_source_prob=target_source_count;
			//TODO print some stuff so we know how we are doing
			System.out.printf("error %f\n ", error);
		}
					
		System.out.printf("Finished training IBM2\n ");
		
	}
	
	private Pair<Integer,Integer> getPairOfInts(int first, int second){
		return new Pair<Integer,Integer>(Integer.valueOf(first ), Integer.valueOf(second ));
	}
}

