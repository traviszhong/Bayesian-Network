package umt.ml.bayesiannet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.StringTokenizer;
/**
 * 
 * @author Zhong Ziyue
 *
 * @email zhongzy@strongit.com.cn
 * 
 * Apr 17, 2016
 */
public class Classifier {
	private int[][] data;
	private int[] label;
	private String dataPath;
	private Map<String,Integer> map;
	private Map<Integer,String> remap;
	private int[][][] bin;
	//order: dimension, class,value on this dim
	
	/**
	 * The constructor, initialize everything and read in the data here
	 */
	public Classifier(String path) throws IOException{
		this.dataPath=path;
		data=new int[1000][5];
		label=new int[1000];
		map=new HashMap<String,Integer>();
		map.put("0",0);
		map.put("1",1);
		remap=new HashMap<Integer,String>();
		remap.put(0,"0");
		remap.put(1,"1");
		bin=new int[5][2][2];
		loadData();
	}
	/**
	 * This function for read in the data
	 * @throws IOException
	 */
	public void loadData() throws IOException{
		File csvData= new File(this.dataPath);
		BufferedReader br=new BufferedReader(new FileReader(csvData));
		String line="";
		int index=-1;
		while((line=br.readLine())!=null){
			if(index==-1){
				index++;continue;
			}
			//Use a tokenizer to split each line of data from the csv file
			StringTokenizer st=new StringTokenizer(line,",");
			for(int i=0;i<=4;i++){
				data[index][i]=map.get(st.nextToken());
			}
			label[index]=map.get(st.nextToken());
			index++;
		}
		br.close();
	}
	/**
	 * calculate the factorial result in a logarithmic way
	 * @param n : calculate the factorial of n
	 * @return
	 */
	public double logFact(int n){
		double rst=0;//log1=0, so the start value is 0
		for(int i=2;i<=n;i++) rst+=(Math.log(i)/Math.log(2));
		return rst;
	}
	/**
	 * This function returns all the combinatorial result from the parent list which serves as 
	 * a cartesian function mentioned in the class
	 * @param parents
	 * @return
	 */
	public List<List<Integer>> parentInst(List<Integer> parents){
		//Basically, this function is implemented by using a Breath First Search
		//backing track algorithm
		List<List<Integer>> rst=new ArrayList<List<Integer>>();
		rst.add(new ArrayList<Integer>());
		//I use a queue to store the interim result
		Queue<List<Integer>> q=new LinkedList<List<Integer>>();
		//put a empty list into the queue for initiallization
		q.offer(new ArrayList<Integer>());
		while(!q.isEmpty()){
			//get a interim result from the queue
			List<Integer> t=q.poll();
			//check each node
			for(int i:parents){
				//if it's already in the list, just skit this one
				if(t.contains(i)||t.size()>0&&t.get(t.size()-1)>i) continue;
				//add this one to the list and push it back to the queue
				t.add(i);
				rst.add(new ArrayList<Integer>(t));
				q.offer(new ArrayList<Integer>(t));
				t.remove(t.size()-1);
			}
		}
		rst.remove(0);
		return rst;
	}
	/**
	 * This is a function for binning
	 * @param trainingData : the selected training dataset, 
	 * in the list it stores the index of row in the original set
	 */
	public void bining(List<Integer> trainingData){
		for(int i:trainingData){
			for(int j=0;j<=4;j++){
				//go through each dimension
				bin[j][label[i]][data[i][j]]++;
			}
		}
	}
	/**
	 * This function returns the score of a specific parent list on a dim
	 * @param index : the index of a dimension
	 * @param parents : a list of parent
	 * @return : the score according to the g algorithm
	 */
	public double g(int index,List<Integer> parents){
		double score=0;
		//for a empty parent list
		if(parents.size()==0){
			//the size of the whole data set
			int nij=900;
			//calculate the first part in the formular, since ri-1=2 all the time in this data set
			//so I simplify it as the follow
			score-=logFact(nij+1);
			//for the two possible value on this dimension
			for(int i=0;i<=1;i++){
				int nijk=0;
				for(int j=0;j<nij;j++){
					if(data[j][index]==i) nijk++;
				}
				score+=logFact(nijk);
			}
		}else{//for the situation which the parent list is not empty
			List<List<Integer>> plist=parentInst(parents);
			for(List<Integer> cp:plist){
				List<Integer> sv=new ArrayList<Integer>();
				//new dataset that have the same value on dimensions in the parent list 
				for(int i=0;i<900;i++){
					//for each record in the dataset
					boolean flag=true;
					//for each node in the parent list
					for(int dim:cp){
						//if it has different value on this dimension with the test sample,
						//then just skip this record
						if(data[i][dim]!=data[index][dim]){
							flag=false;
							break;
						}
					}
					if(flag) sv.add(i);
				}
				int nij=sv.size();
				score-=logFact(nij+1);
				//for each possible value on this dimension
				for(int i=0;i<2;i++){
					int nijk=0;
					for(int s:sv){
						if(data[s][index]==i) nijk++;
					}
					score+=logFact(nijk);
				}
			}
		}
		return score;
	}
	/**
	 * This function implement the k2 algorithm
	 * @return : the parents list for all dimensions on the training set
	 */
	public List<List<Integer>> k2(){
		//make a list for the final result
		List<List<Integer>> rst=new ArrayList<List<Integer>>();
		for(int i=0;i<5;i++){
			//for each dim, I create a list to store the parent
			List<Integer> plist=new ArrayList<Integer>();
			//calculate the g score for current parent list
			double pold=g(i,plist);
			boolean okToProceed=true;
			while(okToProceed&&plist.size()<i){
				double max=-999999;//set the max to minimum at first
				List<Integer> maxList=plist;
				//each node can only make those previous nodes as its parent
				for(int j=0;j<i;j++){
					//if the node is already in the parent list, just skip it
					if(plist.contains(j)) continue;
					List<Integer> newPlist=new ArrayList<Integer>(plist);
					newPlist.add(j);
					double pnew=g(i,newPlist);
					//try to find the node which can get a best g score when adding to the current parent list
					if(pnew>max){
						max=pnew;
						maxList=newPlist;
					}
				}
				//if the new parent list can improve the g score, we use the new one
				if(max>pold){
					pold=max;
					plist=maxList;
				}else okToProceed=false;
			}
			Collections.sort(plist);
			rst.add(plist);
		}
		return rst;
	}
	/**
	 * This function used for naive baysien 
	 * @param sample
	 * @return
	 */
	public int classify(int[] sample){
		double p0=1,p1=1;
		for(int i=0;i<=4;i++){
			int total=bin[i][0][sample[i]]+bin[i][1][sample[i]];
			double zero=bin[i][0][sample[i]];
			double one=bin[i][1][sample[i]];
			if(zero==0) p0*=1.0/1000;
			else p0*=zero/total;
			if(one==0) p1*=1.0/1000;
			else p1*=one/total;
		}
		return p0>p1?0:1;
	}
	/**
	 * This function shrink the data set by using the parent information
	 * @param sample : the sample in test set
	 * @param parents : the parent list
	 * @return : a list of index which is the row number in original data set
	 */
	public List<Integer> shrinkDataSet(int[] sample,List<Integer> parents){
		List<Integer> rst=new ArrayList<Integer>();
		for(int i=0;i<900;i++){
			boolean flag=true;
			for(int p:parents){
				if(data[i][p]!=sample[p]){
					//if the curent sample has different value on a dim in the parent list
					//I won't add it to the new data set
					flag=false;
					break;
				}
			}
			if(flag) rst.add(i);
		}
		return rst;
	}
	/**
	 * This function classification by using baysien network
	 * @param sample
	 * @return
	 */
	public int doClassify(int[] sample){
		//get the parent information for all dims by using k2 algorithm
		List<List<Integer>> parents=k2();
		//set the initial possibility to 1
		double p0=1,p1=1;
		for(int i=0;i<=4;i++){
			List<Integer> p=parents.get(i);
			//shrink and create a new data set
			List<Integer> trainingSet=shrinkDataSet(sample,p);
			bin=new int[5][2][2];
			//do the binning by using the new dataset
			bining(trainingSet);
			int total=bin[i][0][sample[i]]+bin[i][1][sample[i]];
			double zero=bin[i][0][sample[i]];
			double one=bin[i][1][sample[i]];
			if(zero==0) p0*=1.0/1000;
			else p0*=zero/total;
			if(one==0) p1*=1.0/1000;
			else p1*=one/total;
		}
		return p0>p1?0:1;
	}
	/**
	 * This function swap the position of two record in the data set
	 * @param i
	 * @param j
	 */
	public void swap(int i,int j){
		int[] t=data[i];
		data[i]=data[j];
		data[j]=t;
		int tl=label[i];
		label[i]=label[j];
		label[j]=tl;
	}
	/**
	 * This function do the 10 folders cross validation
	 */
	public void crossValidation(){
		//use a variable to store the sum of the 10 accuracy
		double sum=0;
		sum+=test();
		int base=0;//set the base to 0 at first
		for(int i=0;i<100;i++) swap(base+i,900+i);//swap the data in database, create a new folder
		sum+=test();//get the result of current folder
		base+=100;
		for(int i=0;i<100;i++) swap(base+i,900+i);//swap the data in database, create a new folder
		sum+=test();//get the result of current folder
		base+=100;
		for(int i=0;i<100;i++) swap(base+i,900+i);//swap the data in database, create a new folder
		sum+=test();//get the result of current folder
		base+=100;
		for(int i=0;i<100;i++) swap(base+i,900+i);//swap the data in database, create a new folder
		sum+=test();//get the result of current folder
		base+=100;
		for(int i=0;i<100;i++) swap(base+i,900+i);//swap the data in database, create a new folder
		sum+=test();//get the result of current folder
		base+=100;
		for(int i=0;i<100;i++) swap(base+i,900+i);//swap the data in database, create a new folder
		sum+=test();//get the result of current folder
		base+=100;
		for(int i=0;i<100;i++) swap(base+i,900+i);//swap the data in database, create a new folder
		sum+=test();//get the result of current folder
		base+=100;
		for(int i=0;i<100;i++) swap(base+i,900+i);//swap the data in database, create a new folder
		sum+=test();//get the result of current folder
		base+=100;
		for(int i=0;i<100;i++) swap(base+i,900+i);//swap the data in database, create a new folder
		sum+=test();//get the result of current folder
		base+=100;
		System.out.println("The Average Accuracy:"+sum/10);
	}
	public double test(){
		int count=0;
		int truepos=0,trueneg=0,falsepos=0,falseneg=0;
		//go through all the samples in test data area
		for(int i=900;i<1000;i++){
			int rst=doClassify(data[i]);
			if(rst==label[i]){
				count++;
				if(rst==0) trueneg++;//accumulate the true negative
				else truepos++;//true positive
			}else{
				if(rst==0) falseneg++;
				else falsepos++;
			}
		}
		//print to the console
		System.out.println("INFO: Accuracy: "+count/100.0+" TP: "+truepos+" TN: "+trueneg+" FP: "+falsepos+" FN: "+falseneg);
		return count/100.0;
	}
}
