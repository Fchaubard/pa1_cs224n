package cs224n.wordaligner;  


import cs224n.util.*;

import java.util.List;
import java.util.Random;

/** 
 * IBM2 word alignment model 
 * 
 * @author Francois Chaubard, Colin Mayer
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

	public void train(List<SentencePair> trainingPairs) {
		//Initialize T(s|t) with IBM Model 1	
		source_target_prob= IBM1.buildSTP(trainingPairs);
		System.out.printf("Finished with Model 1 \n" );
		
		// setup for Model 2
		l_m_i_j_qML = new CounterMap<Pair<Integer,Integer>,Pair<Integer,Integer>>(); // c(f|e)
		source_target_tML = new CounterMap<String,String>();
		
		// init t to probs from model 1
		source_target_tML = source_target_prob;

		CounterMap<String,String> source_target_count;
		// init l_m_i_j_count randomly
		CounterMap<Pair<Integer,Integer>,Pair<Integer,Integer>> l_m_i_j_count = new CounterMap<Pair<Integer,Integer>,Pair<Integer,Integer>>(); // c(f|e)
		CounterMap<Pair<Integer,Integer>,Integer> l_m_i_count = new CounterMap<Pair<Integer,Integer>,Integer>(); // c(i,l,m), TargLength, SourceLegnth, SourceIdx
		
		System.out.printf("Starting Model 2 \n" );
		for(SentencePair pair : trainingPairs){
			int numSourceWords = pair.getSourceWords().size();
			int numTargetWords = pair.getTargetWords().size();
				
			for (int targetIdx = 0; targetIdx < numTargetWords; targetIdx++) {
				
				for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
					
					// initially set count to be random numbers
					l_m_i_j_qML.setCount(getPairOfInts(numTargetWords,numSourceWords), getPairOfInts(srcIndex, targetIdx ),1.0f / numSourceWords);
					
				}
			}
		}
		
		//Convergence criteria
		boolean converged = false;
		int count=0;
		
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
				for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
					// find delta = ... 
					double delta_denominator_sum = 0.0;
					// sum over the target words
					for (int targ = 0; targ < numTargetWords; targ++) {
						delta_denominator_sum += source_target_tML.getCount( pair.getSourceWords().get(srcIndex),pair.getTargetWords().get(targ) );
					}

					for (int targetIdx = 0; targetIdx < numTargetWords; targetIdx++) {
						String target = pair.getTargetWords().get(targetIdx);

						double delta_ijlm =  source_target_tML.getCount( pair.getSourceWords().get(srcIndex), target  ) / delta_denominator_sum;

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
			double error1=0.0;
			double error2 =0.0;
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
			
			if (((error1+error2) < 0.4) | (count > 40)){
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

