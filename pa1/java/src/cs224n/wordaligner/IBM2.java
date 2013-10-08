package cs224n.wordaligner;  

import cs224n.util.*;
import java.util.List;

/**
 * IBM2 models the problem 
 * 
 * @author Francois Chaubard, Colin Mayer
 */

public class IBM2 implements WordAligner {

	// Count of sentence pairs that contain source and target words
	private CounterMap<String,String> source_target_prob; // t(e|f)
	private CounterMap<Pair<Pair<Integer,Integer>,Integer>,Integer> l_m_i_j_qML; // q(j|i,m,l) 

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
				double ai = l_m_i_j_qML.getCount(getIntegerTriple(numSourceWords, numTargetWords, targetIdx), srcIndex) *
						source_target_prob.getCount(source, target);
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
		//Initialize t(f|e) from IBM1 model
		source_target_prob = IBM1.buildSTP(trainingPairs);
		System.out.printf("Finished with Model 1 \n" );

		// setup for Model 2
		l_m_i_j_qML = new CounterMap<Pair<Pair<Integer,Integer>,Integer>,Integer>(); // q(j|i, l, m) 

		// init l_m_i_j_count randomly
		CounterMap<Pair<Pair<Integer,Integer>,Integer>, Integer> l_m_i_j_count = new CounterMap<Pair<Pair<Integer,Integer>,Integer>, Integer>(); // c(j|i,l,m)
		//Counter<Pair<Pair<Integer,Integer>,Integer>> l_m_i_count = new Counter<Pair<Pair<Integer,Integer>,Integer>>(); // c(i,l,m)

		System.out.printf("Starting Model 2 \n" );
		for(SentencePair pair : trainingPairs){
			int numSourceWords = pair.getSourceWords().size();
			int numTargetWords = pair.getTargetWords().size();
			for (int targetIdx = 0; targetIdx < numTargetWords; targetIdx++) {
				for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
					// initially set count to be random numbers
					l_m_i_j_qML.setCount(getIntegerTriple(numSourceWords,numTargetWords,targetIdx), srcIndex, 1.0);
				}
			}
		}	
		l_m_i_j_qML = Counters.conditionalNormalize(l_m_i_j_qML);
		System.out.printf("Finished init for Model 2 \n" );

		// end of init
		boolean converged = false;
		int count=0;

		// begin training Model 2
		while (!converged){
			count++;

			// target_source_count = 0s
			CounterMap<String,String> source_target_count = new CounterMap<String,String>(); // c(e,f)

			// l_m_i_j_count = 0s
			l_m_i_j_count = new CounterMap<Pair<Pair<Integer,Integer>,Integer>, Integer>();
			//l_m_i_count = new Counter<Pair<Pair<Integer,Integer>,Integer>>();

			for(SentencePair pair : trainingPairs){
				int numSourceWords = pair.getSourceWords().size();
				int numTargetWords = pair.getTargetWords().size();
				for (int targetIdx = 0; targetIdx < numTargetWords; targetIdx++) {
					// find delta = ... 
					double delta_denominator_sum = 0.0;
					// sum over the target words
					for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
						delta_denominator_sum += l_m_i_j_qML.getCount(getIntegerTriple(numSourceWords, numTargetWords, targetIdx),srcIndex) *
								source_target_prob.getCount(pair.getSourceWords().get(srcIndex),pair.getTargetWords().get(targetIdx));
					}
					for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
						double delta_ijlm = l_m_i_j_qML.getCount(getIntegerTriple(numSourceWords, numTargetWords, targetIdx),srcIndex) *
								source_target_prob.getCount(pair.getSourceWords().get(srcIndex),pair.getTargetWords().get(targetIdx)) / 
								delta_denominator_sum;

						// add delta to the counters
						l_m_i_j_count.incrementCount(getIntegerTriple(numSourceWords, numTargetWords, targetIdx),srcIndex, delta_ijlm);
						//l_m_i_count.incrementCount(getIntegerTriple(numSourceWords, numTargetWords,targetIdx), delta_ijlm);
						source_target_count.incrementCount(pair.getSourceWords().get(srcIndex), pair.getTargetWords().get(targetIdx), delta_ijlm);
					}

				}

			}
			//normalize count to become probs .. we dont set it directly to tML because we want to check for convergence.. we will do it later
			source_target_count = Counters.conditionalNormalize(source_target_count);
			l_m_i_j_count = Counters.conditionalNormalize(l_m_i_j_count);
			
			// do we need to represent c(jilm) and c(ilm) seperately as well do we need to rep c(f,e) and c(e) separately?
			double error1=0.0;
			double error2 =0.0;
			int numSourceWords,numTargetWords;
			String source,target;
			for(SentencePair pair : trainingPairs){
				numSourceWords = pair.getSourceWords().size();
				numTargetWords = pair.getTargetWords().size();
				for (int targetIdx = 0; targetIdx < numTargetWords; targetIdx++) {
					for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
						source = pair.getSourceWords().get(srcIndex);
						target = pair.getTargetWords().get(targetIdx);
						error1 += Math.pow(source_target_prob.getCount(source, target) - source_target_count.getCount(source, target) ,2);
						// summing over target indexes (english aka j)
						error2 += Math.pow(l_m_i_j_qML.getCount(getIntegerTriple(numSourceWords, numTargetWords, targetIdx), srcIndex)   -   l_m_i_j_count.getCount(getIntegerTriple(numSourceWords, numTargetWords, targetIdx),srcIndex),2);// / l_m_i_count.getCount(getIntegerTriple(numSourceWords, numTargetWords, targetIdx))  ,2);
						l_m_i_j_qML.setCount(getIntegerTriple(numSourceWords, numTargetWords, targetIdx), srcIndex, l_m_i_j_count.getCount(getIntegerTriple(numSourceWords, numTargetWords, targetIdx),srcIndex));// / l_m_i_count.getCount(getIntegerTriple(numSourceWords, numTargetWords, targetIdx)));
					}
				}
			}

			if (((error1+error2) < 0.5) | (count > 50)){
				converged=true;
			}
			//TODO print some stuff so we know how we are doing
			System.out.printf("%d error1 %f error2 %f\n ",count, error1,error2);

			// update tML
			source_target_prob = source_target_count;
			l_m_i_j_qML = l_m_i_j_count;
			
		}// while converge loop

		System.out.printf("Finished training IBM2\n ");

	}
	
	private Pair<Pair<Integer,Integer>,Integer> getIntegerTriple(int first, int second, int third){
		Pair<Integer,Integer> p1 = new Pair(first,second);
		return (new Pair(p1,third));
	}
}

