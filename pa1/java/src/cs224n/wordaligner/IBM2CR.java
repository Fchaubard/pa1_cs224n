package cs224n.wordaligner;


import cs224n.util.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** 
 * IBM2CR models the problem 
 * 
 * @author Francois Chaubard
 */


public class IBM2CR implements WordAligner {

	// Count of sentence pairs that contain source and target words
	private CounterMap<String,String> target_source_t; // t(ei|fj)
	private CounterMap<Integer,Integer> source_target_d; // d(i|j) 
	
	// Count of source words appearing in the training set
	//private Counter<String> source_count;

	@Override
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
				double ai = source_target_d.getCount(targetIdx,srcIndex) * target_source_t.getCount(source,target);

				if (currentMax < ai){
					currentMax = ai;
					maxSourceIdx = srcIndex;
				}
			}
			
			alignment.addPredictedAlignment(targetIdx, maxSourceIdx);
		}
		return alignment;
	}

	@Override
	public void train(List<SentencePair> trainingPairs) {
		//Test params
		int S=7; //num iterations
		double lambda=0.01; //regularization
		double gamma=0.5; //step size
		int B=100; //batch size
		int numSentences = trainingPairs.size();
		
		//Initalize counters
		target_source_t= new CounterMap<String,String>(); // t(ei|fj)
		source_target_d= new CounterMap<Integer,Integer>(); // d(i|j) 

		//HashSet<String> source_dict = new HashSet<String>();
		//HashSet<String> target_dict = new HashSet<String>();
		HashMap<String,Set<String>> target_D = new HashMap<String,Set<String>>(); // c(f|e)
		int M=0;
		int L=0;
		
		System.out.printf("Starting init for Model 2 Convex \n" );
		
		// Find M, L, E, F, and D(e)
		for(SentencePair pair : trainingPairs){
			// add to target sentence? it says to do so ... but doesnt make sense
			pair.sourceWords.add(NULL_WORD);
			int numSourceWords = pair.getSourceWords().size();
			int numTargetWords = pair.getTargetWords().size();
			M=Math.max(numTargetWords, M);
			L=Math.max(numSourceWords, L);
			
			for (int targetIdx = 0; targetIdx < numTargetWords; targetIdx++) {
				String target = pair.getTargetWords().get(targetIdx);
				Set<String> set = new HashSet<String>();
				target_D.put(target , set);
				for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
					String source = pair.getSourceWords().get(srcIndex);
					//source_dict.add(source);
					//target_dict.add(target);
					target_D.get(target).add(source);
				}
			}
		}
		
		// init t and d
		/*for(SentencePair pair : trainingPairs){
			int numSourceWords = pair.getSourceWords().size();
			int numTargetWords = pair.getTargetWords().size();
			
			for (int targetIdx = 0; targetIdx < numTargetWords; targetIdx++) {
				String target = pair.getTargetWords().get(targetIdx);
				for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
					String source = pair.getSourceWords().get(srcIndex);
					target_source_t.setCount(target, source, 1.0/target_D.get(target).size() );
					source_target_d.setCount(srcIndex, targetIdx, 1.0/(1+L));
				}
			}
		}*/
		
		// init t
		for(SentencePair pair : trainingPairs){
			for (String source : pair.getSourceWords()) {
				for (String target : pair.getTargetWords()) {
					target_source_t.setCount(source, target, 1.0f/target_D.get(target).size() );
				}
			}
		}
		
		// init d
		for (int targetIdx = 0; targetIdx < M; targetIdx++) {
			for (int srcIndex = 0; srcIndex < L; srcIndex++) {
				source_target_d.setCount( targetIdx, srcIndex, 1.0f/(1+L));
			}
		}
		
		System.out.printf("Finished init for Model 2 Convex \n" );
		int K=(int) Math.floor( numSentences/B );
		for (int s = 0; s < S; s++) {
			
			for (int b = 1; b < K; b++) {
				
				int mk =0;
				int lk =0;
				CounterMap<String,String> target_source_alpha = new CounterMap<String,String>(); // a(ei|fj)
				CounterMap<Integer,Integer> source_target_beta= new CounterMap<Integer,Integer>(); // Beta(i|j) 
				
				int sentence_begin=b*K-K;
				for (int sentenceIdx = sentence_begin; sentenceIdx < (sentence_begin+K); sentenceIdx++) {
					
					SentencePair pair = trainingPairs.get(sentenceIdx);
					int numSourceWords = pair.getSourceWords().size();
					int numTargetWords = pair.getTargetWords().size();
					
					mk=Math.max(numTargetWords, mk);
					lk=Math.max(numSourceWords, lk);
					
					for (int targetIdx = 0; targetIdx < numTargetWords; targetIdx++) {
						String target = pair.getTargetWords().get(targetIdx);
						for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
							String source = pair.getSourceWords().get(srcIndex);
							
							double R_denominator = 0;
							double Q_denominator = 0;
							
							for (int targ = 0; targ < numTargetWords; targ++){
								R_denominator += target_source_t.getCount( source,pair.getTargetWords().get(targ));
								Q_denominator += Math.min(target_source_t.getCount( source,pair.getTargetWords().get(targ)) , source_target_d.getCount(targ,srcIndex));	
							}
							
							double R = 1.0f / (2*(lambda + R_denominator));
							double Q = 1.0f / (2*(lambda + Q_denominator));
							
							target_source_alpha.incrementCount( source,target, R);
							
							if(target_source_t.getCount(source,target) <= source_target_d.getCount(targetIdx,srcIndex) ){
								target_source_alpha.incrementCount( source,target, Q);
							}else{
								source_target_beta.incrementCount( targetIdx, srcIndex,Q);
							}
							
						}//source words
					}//target words
					
					
				}// subset of sentences 
				
				// increment t
				sentence_begin=b*K-K;
				for (int sentenceIdx = sentence_begin; sentenceIdx < (sentence_begin+K); sentenceIdx++) {
					
					SentencePair pair = trainingPairs.get(sentenceIdx);
					// add to target sentence? it says to do so ... but doesnt make sense
					int numSourceWords = pair.getSourceWords().size();
					int numTargetWords = pair.getTargetWords().size();
					
					for (int targetIdx = 0; targetIdx < numTargetWords; targetIdx++) {
						String target = pair.getTargetWords().get(targetIdx);
						for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
							String source = pair.getSourceWords().get(srcIndex);
							target_source_t.setCount( source, target, target_source_t.getCount( source,target)*Math.exp(gamma*target_source_alpha.getCount(source,target) /B) );
							source_target_d.setCount( targetIdx,srcIndex, source_target_d.getCount( targetIdx,srcIndex)*Math.exp(gamma*source_target_beta.getCount(targetIdx,srcIndex) /B) );
							
						}
					}
				}
					
				
				target_source_t = Counters.conditionalNormalize(target_source_t);
				source_target_d = Counters.conditionalNormalize(source_target_d);
				
				System.out.printf("step=%d b=%d\n",s,b);
			}// subset
		}//iterations
		System.out.printf("Finished training Model 2 Convex \n" );
		
	}

}

//java -cp ~/cs224n/pa1_cs224n/pa1/java/classes cs224n.assignments.WordAlignmentTester -dataPath /afs/ir/class/cs224n/pa1/data/ -model cs224n.wordaligner.IBM2CR -evalSet dev -trainSentences 5000