package cs224n.wordaligner;  


import cs224n.util.*;

import java.util.List;
import java.util.Random;

/** 
 * IBM2 models the problem 
 * 
 * @author Francois Chaubard
 */

public class IBM2 implements WordAligner {

	// Count of sentence pairs that contain source and target words
	private CounterMap<String,String> source_target_prob; // p(f|e)
	private CounterMap<String,String> source_target_tML; // t(fi|ej)
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
				double ai = l_m_i_j_qML.getCount(getPairOfInts(numTargetWords,numSourceWords),getPairOfInts(srcIndex,targetIdx)) * source_target_tML.getCount(source, target) ;

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
			
			
		}// while for Model 1
		
		System.out.printf("Finished with Model 1 \n" );
		
		
		
		
		
		// setup for Model 2
		l_m_i_j_qML = new CounterMap<Pair<Integer,Integer>,Pair<Integer,Integer>>(); // c(f|e)
		source_target_tML = new CounterMap<String,String>();
		
		// init t to probs from model 1
		source_target_tML = source_target_prob;
		
		// init l_m_i_j_count randomly
		CounterMap<Pair<Integer,Integer>,Pair<Integer,Integer>> l_m_i_j_count = new CounterMap<Pair<Integer,Integer>,Pair<Integer,Integer>>(); // c(f|e)
		CounterMap<Pair<Integer,Integer>,Integer> l_m_i_count = new CounterMap<Pair<Integer,Integer>,Integer>(); // c(i,l,m), TargLength, SourceLegnth, SourceIdx
		
		System.out.printf("Starting Model 2 \n" );
		Random randomGenerator = new Random();
		for(SentencePair pair : trainingPairs){
			int numSourceWords = pair.getSourceWords().size();
			int numTargetWords = pair.getTargetWords().size();
				
			for (int targetIdx = 0; targetIdx < numTargetWords; targetIdx++) {
				
				for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
					
					// initially set count to be random numbers
					l_m_i_j_qML.setCount(getPairOfInts(numTargetWords,numSourceWords), getPairOfInts(srcIndex, targetIdx ), 1.0f /numSourceWords );
					
				}
				
				//l_m_i_j_count.incrementCount(getPairOfInts(numTargetWords,numSourceWords), getPairOfInts(maxSourceIdx, targetIdx ),1);
			}
		}// sentences
		
		// now init qML randomly by normalizing out l_m_i_j_count
		/*
		for(Pair<Integer,Integer> key1: l_m_i_j_count.keySet()){
			
			for (Pair<Integer,Integer> key2: l_m_i_j_count.getCounter(key1).keySet()){
				
				double counter_ilm = 0;
				// summing over target indexes (english aka j)
				for (Pair<Integer,Integer> key2_sum_out: l_m_i_j_count.getCounter(key1).keySet()){
					counter_ilm += l_m_i_j_count.getCount(key1,getPairOfInts(key2.getFirst(), key2_sum_out.getSecond()) );
				}
				l_m_i_count.setCount(key1,key2.getFirst(),counter_ilm);
				l_m_i_j_qML.setCount(key1, key2, l_m_i_j_count.getCount(key1, key2) / counter_ilm);
			}
			
		}// sentences
		
		*/
		System.out.printf("Finished init for Model 2 \n" );
		
		// end of init
		
		
		
		
		converged = false;
		count=0;
		
		// begin training Model 2
		while (!converged){
			count++;
			
			// source_target_count = 0s
			source_target_count = new CounterMap<String,String>(); // c(f|e)
			
			// l_m_i_j_count = 0s
			l_m_i_j_count = new CounterMap<Pair<Integer,Integer>,Pair<Integer,Integer>>(); // c(j|i m l) TargLength, SourceLength, SourceIdx, TargetIdx
			l_m_i_count = new CounterMap<Pair<Integer,Integer>,Integer>(); // c(i,l,m), TargLength, SourceLegnth, SourceIdx
			
			for(SentencePair pair : trainingPairs){
				int numSourceWords = pair.getSourceWords().size();
				int numTargetWords = pair.getTargetWords().size();
				for (int targetIdx = 0; targetIdx < numTargetWords; targetIdx++) {
					String target = pair.getTargetWords().get(targetIdx);
					
					for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
						// find delta = ... 
						double delta_denominator_sum = 0.0;
						// sum over the target words
						for (int src = 0; src < numSourceWords; src++) {
							delta_denominator_sum +=    l_m_i_j_qML.getCount(getPairOfInts(pair.getTargetWords().size(), pair.getSourceWords().size()), getPairOfInts(src,targetIdx))* 
									source_target_tML.getCount( pair.getSourceWords().get(src), pair.getTargetWords().get(targetIdx) );
						}
						
						double delta_ijlm =  l_m_i_j_qML.getCount(getPairOfInts(pair.getTargetWords().size(),pair.getSourceWords().size()), getPairOfInts(srcIndex,targetIdx))*source_target_tML.getCount( pair.getSourceWords().get(srcIndex), target  ) / delta_denominator_sum;
						
						// add delta to the two counters
						l_m_i_j_count.incrementCount(getPairOfInts(numTargetWords, numSourceWords ),    
								                     getPairOfInts(srcIndex, targetIdx ), delta_ijlm);
						l_m_i_count.incrementCount(getPairOfInts(numTargetWords, numSourceWords ),    
			                     					Integer.valueOf(srcIndex), delta_ijlm);
						
						source_target_count.incrementCount(pair.getSourceWords().get(srcIndex),pair.getTargetWords().get(targetIdx),delta_ijlm);
						
						//System.out.printf(" denom %f  delta %f  tML %f \n ", delta_denominator_sum, delta_ijlm, source_target_tML.getCount( target, pair.getSourceWords().get(srcIndex)) );
						
					}
					
				}
				
			}// sentences
			
			//normalize count to become probs .. we dont set it directly to tML because we want to check for convergence.. we will do it later
			source_target_count = Counters.conditionalNormalize(source_target_count);
			
			// do we need to represent c(jilm) and c(ilm) seperately as well do we need to rep c(f,e) and c(e) separately?
			double error1 = 0.0;
			double error2 = 0.0;
			for(SentencePair pair : trainingPairs){
				List<String> targetWords = pair.getTargetWords();
				List<String> sourceWords = pair.getSourceWords();
				for(String source : sourceWords){
					for(String target : targetWords){
						error1 += Math.pow(source_target_tML.getCount(source, target) - source_target_count.getCount(source, target) ,2);
					}
				}
			}
			for(Pair<Integer,Integer> key1: l_m_i_j_count.keySet()){
				
				for (Pair<Integer,Integer> key2: l_m_i_j_count.getCounter(key1).keySet()){
					
					// summing over target indexes (english aka j)
					error2 += Math.pow(  l_m_i_j_qML.getCount(key1, key2)   -   l_m_i_j_count.getCount(key1, key2) / l_m_i_count.getCount(key1,key2.getFirst())  ,2);
				}
				
			}// sentences
			
			if (((error1+error2) < 0.04) | (count > 50)){
				converged=true;
			}
			
			//TODO print some stuff so we know how we are doing
			System.out.printf("%d error1 %f error2 %f\n ",count, error1,error2);
			
			// update tML
			source_target_tML = source_target_count;
			
			// update qML
			for(Pair<Integer,Integer> key1: l_m_i_j_count.keySet()){
				
				for (Pair<Integer,Integer> key2: l_m_i_j_count.getCounter(key1).keySet()){
					
					// summing over target indexes (english aka j)
					l_m_i_j_qML.setCount(key1, key2, l_m_i_j_count.getCount(key1, key2) / l_m_i_count.getCount(key1,key2.getFirst()) );
				}
				
			}// sentences
		}// while converge loop
					
		System.out.printf("Finished training IBM2\n ");
		
	}
	
	private Pair<Integer,Integer> getPairOfInts(int first, int second){
		return new Pair<Integer,Integer>(Integer.valueOf(first ), Integer.valueOf(second ));
	}
}

