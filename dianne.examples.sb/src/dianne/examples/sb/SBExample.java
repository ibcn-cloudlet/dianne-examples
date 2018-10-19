package dianne.examples.sb;

import java.io.IOException;
import java.util.Random;

import javax.servlet.AsyncContext;
import javax.servlet.AsyncEvent;
import javax.servlet.AsyncListener;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.osgi.service.component.annotations.Activate;
import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Deactivate;
import org.osgi.service.component.annotations.Reference;
import org.osgi.service.http.HttpService;
import org.osgi.util.pushstream.PushStream;

import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

import be.iminds.iot.dianne.api.nn.Dianne;
import be.iminds.iot.dianne.api.nn.NeuralNetwork;
import be.iminds.iot.dianne.api.nn.platform.DiannePlatform;
import be.iminds.iot.dianne.tensor.Tensor;
import be.iminds.iot.dianne.tensor.TensorOps;
import be.iminds.iot.dianne.tensor.util.JsonConverter;
import be.iminds.iot.robot.api.arm.Arm;
import be.iminds.iot.robot.api.omni.OmniDirectional;
import be.iminds.iot.sensor.api.LaserScan;
import be.iminds.iot.sensor.api.LaserScanner;
import be.iminds.iot.sensor.util.SensorStreams;

@Component(service={javax.servlet.Servlet.class},
property={"alias:String=/example/sb",
	 	  "osgi.http.whiteboard.servlet.pattern=/example/sbsse",
	 	  "osgi.http.whiteboard.servlet.asyncSupported:Boolean=true",
		  "aiolos.proxy=false"},
immediate=true)
public class SBExample extends HttpServlet {

	private Arm arm;
	private OmniDirectional base;
	private LaserScanner lidar;
	private SensorStreams sensors;
	private PushStream<LaserScan> stream;

	private DiannePlatform diannePlatform;
	private Dianne dianne;
	
	private NeuralNetwork prior;
	private NeuralNetwork posterior;
	private NeuralNetwork likelihood;
	private NeuralNetwork policy;
	
	private Tensor state;
	private Tensor action;
	private int a = -1;
	private Tensor q;
	private Tensor observation;
	private Tensor priorDistribution;
	private Tensor posteriorDistribution;
	private Tensor reconstruction;
	
	private Random random = new Random();
	private float speed = 0.2f;
	
	private AsyncContext async;
	private JsonConverter jsonConverter = new JsonConverter();
	private int skip = 4;
	private volatile boolean evalPolicy = false;
	
	private String task = "docking";
	
	void process(LaserScan lidar) {
		try {
			if(observation == null) {
				observation = new Tensor(lidar.data, lidar.data.length);
			} else {
				observation.set(lidar.data);
			}
			
			// estimate current state, given observation
			posteriorDistribution = posterior.forward(new String[] {"State","Action","Observation"}, new String[]{"Output"}, new Tensor[] {state, action, observation}).getValue().tensor;
			
			// TODO calculate KL prior - posterior?
			state = sampleFromGaussian(state, posteriorDistribution);
			float kl = kl(posteriorDistribution, priorDistribution);
			
			// reconstruct
			reconstruction = likelihood.forward(state).narrow(0, observation.size());
			
			// evaluate policy
			q = policy.forward(state);
			if(evalPolicy) {
				a = TensorOps.argmax(q);
			}
			execute(a);
			action.fill(0.0f);
			if(a > 0 && a < action.size()) {
				action.set(1.0f, a);
			}

			// log over SSE
			log(observation, priorDistribution, posteriorDistribution, reconstruction, q, kl);

			// estimate next state
			priorDistribution = prior.forward(new String[] {"State","Action"}, new String[] {"Output"}, new Tensor[] {state, action}).getValue().tensor;

		} catch(Throwable t) {
			t.printStackTrace();
		}
	}
	
	void execute(int action) {
		switch(action){
		case 0:
			base.move(0f, speed, 0f);
			break;
		case 1:
			base.move(0f, -speed, 0f);
			break;
		case 2:
			base.move(speed, 0f, 0f);
			break;
		case 3:
			base.move(-speed, 0f, 0f);
			break;
		case 4:
			base.move(0f, 0.f, 2*speed);
			break;
		case 5:
			base.move(0f, 0.f, -2*speed);
			break;	
		case 6:
			base.stop();	

			// only move arm in case of fetching?
//			Promise<Arm> result = arm.openGripper()
//				.then(p -> arm.setPositions(2.92f, 0.0f, 0.0f, 0.0f, 2.875f))
//				.then(p -> arm.setPositions(2.92f, 1.76f, -1.37f, 2.55f))
//				.then(p -> arm.closeGripper())
//				.then(p -> arm.setPositions(0.01f, 0.8f))
//				.then(p -> arm.setPositions(0.01f, 0.8f, -1f, 2.9f))
//				.then(p -> arm.openGripper())
//				.then(p -> arm.setPosition(1, -1.3f))
//				.then(p -> arm.reset());
//			result.getValue();
			break;
		default:
			base.stop();
		}
	}

	private int i=0;
	void log(Tensor observation, Tensor priorDistribution, Tensor posteriorDistribution, Tensor reconstruction, Tensor q, float kl) {
		i++;
		try {
			if(async != null) {
				JsonObject data = new JsonObject();
				if(i % skip == 0) {
					data.add("observation", jsonConverter.toJson(observation));
					data.add("prior", jsonConverter.toJson(priorDistribution));
					data.add("posterior", jsonConverter.toJson(posteriorDistribution));
					data.add("reconstruction", jsonConverter.toJson(reconstruction));
					//data.add("q", jsonConverter.toJson(q));
				}
				data.add("kl", new JsonPrimitive(kl));
				
				StringBuilder builder = new StringBuilder();
				builder.append("data: ").append(data.toString()).append("\n\n");
				async.getResponse().getWriter().print(builder.toString());
				async.getResponse().getWriter().flush();
			}
		} catch(Throwable t) {
			// ignore?
		}
	}
	
	@Activate
	void activate() throws Exception {
		// initialize Tensor variables
		// TODO parameterize sizes?
		state = new Tensor(20);
		state.fill(0.0f);
		
		action = new Tensor(7);
		action.fill(0.0f);
		
		// initialize neural networks for all models
		prior = dianne.getNeuralNetwork(diannePlatform.deployNeuralNetwork("Prior")).getValue();
		posterior = dianne.getNeuralNetwork(diannePlatform.deployNeuralNetwork("Posterior")).getValue();
		likelihood = dianne.getNeuralNetwork(diannePlatform.deployNeuralNetwork("Likelihood")).getValue();
		policy = dianne.getNeuralNetwork(diannePlatform.deployNeuralNetwork("Policy", new String[] {task})).getValue();

		// bootstrap prior distribution
		priorDistribution = prior.forward(new String[] {"State","Action"}, new String[] {"Output"}, new Tensor[] {state, action}).getValue().tensor;
		
		// start processing lidar data
		stream  = sensors.stream(lidar);
	    stream.forEach(this::process);
	}
	
	@Deactivate
	void deactivate() {
	    stream.close();
	    
	    diannePlatform.undeployNeuralNetwork(prior.getId());
	    diannePlatform.undeployNeuralNetwork(posterior.getId());
	    diannePlatform.undeployNeuralNetwork(likelihood.getId());
	    diannePlatform.undeployNeuralNetwork(policy.getId());
	}
	
	private Tensor sampleFromGaussianMixture(Tensor result, Tensor distribution, int batchSize) {
		if(batchSize > 1) {
			for(int i = 0; i < batchSize; i++)
				sampleFromGaussianMixture(result.select(0, i), distribution);
		}
		return result;
	}
	
	private Tensor sampleFromGaussianMixture(Tensor result, Tensor distribution) {
		return sampleFromGaussian(result, distribution.select(0, random.nextInt(distribution.size(0))));
	}
	
	private Tensor sampleFromGaussian(Tensor result, Tensor distribution) {
		int size = distribution.size()/2;
		Tensor means = distribution.narrow(0, 0, size);
		Tensor stdevs = distribution.narrow(0, size, size);
		
		Tensor random = new Tensor(means.size());
		random.randn();
		
		TensorOps.cmul(result, random, stdevs);
		TensorOps.add(result, result, means);
		return result;
	}
	
	public float kl(Tensor output, Tensor target) {
		Tensor l = null;
		Tensor outStdev = null;
		Tensor tarStdev = null;
		Tensor stdevRatio = null;
		Tensor logStdevRatio = null;
		
		float EPS = 1e-6f;
		
		int dim = output.dim()-1;
		int size = output.size(dim)/2;
		
		// loss = log(s_tar / s_out) + (s_out^2 + (mu_out - mu_tar)^2) / (2 * s_tar^2) - 1/2
		//      = ((s_out / s_tar)^2 + ((mu_out - mu_tar) / s_tar)^2 - log((s_out / s_tar)^2) - 1) / 2
		Tensor outMean = output.narrow(dim, 0, size);
		Tensor tarMean = target.narrow(dim, 0, size);
		outStdev = TensorOps.add(outStdev, output.narrow(dim, size, size), EPS);
		tarStdev = TensorOps.add(tarStdev, target.narrow(dim, size, size), EPS);
		
		l = TensorOps.sub(l, outMean, tarMean);
		l = TensorOps.cdiv(l, l, tarStdev);
		TensorOps.cmul(l, l, l);
		
		stdevRatio = TensorOps.cdiv(stdevRatio, outStdev, tarStdev);
		TensorOps.cmul(stdevRatio, stdevRatio, stdevRatio);
		TensorOps.add(l, l, stdevRatio);
		
		logStdevRatio = TensorOps.log(logStdevRatio, stdevRatio);
		TensorOps.sub(l, l, logStdevRatio);
		
		TensorOps.sub(l, l, 1);
		TensorOps.div(l, l, 2);
		
		return TensorOps.sum(l);
	}

	
	@Override
	protected void doGet(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		// write text/eventstream response
		response.setContentType("text/event-stream");
		response.setHeader("Cache-Control", "no-cache");
		response.setCharacterEncoding("UTF-8");
		response.addHeader("Connection", "keep-alive");
	
		// TODO atm only one connection at a time is supported
		async = request.startAsync();
		async.setTimeout(300000); // let it ultimately timeout if client is closed
		
		async.addListener(new AsyncListener() {
			@Override
			public void onTimeout(AsyncEvent e) throws IOException {
				async = null;
			}
			@Override
			public void onStartAsync(AsyncEvent e) throws IOException {
			}
			@Override
			public void onError(AsyncEvent e) throws IOException {
				async = null;
			}
			@Override
			public void onComplete(AsyncEvent e) throws IOException {
				async = null;
			}
		});
	}
	
	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		boolean pressed = request.getParameter("type").equals("keydown");
		String key = request.getParameter("key");
		if(key==null){
			// on chrome you only get code?!
			// use code instead or convert to key or pass both?!
			String code = request.getParameter("code");
			if(code.startsWith("Key")){
				key = code.substring(3).toLowerCase();
			} else if(code.startsWith("Digit")){
				key = code.substring(5).toLowerCase();
			} else {
				key = code;
			}
		}
		
		if(pressed) {
			switch(key){
			case "ArrowUp":
			case "w":
				a = 2;
				break;
			case "ArrowDown":
			case "s":
				a = 3;
				break;
			case "ArrowLeft":
			case "a":
				a=0;
				break;
			case "ArrowRight":
			case "d":
				a=1;
				break;
			case "q":
				a=4;
				break;
			case "e":
				a=5;
				break;
			case "Enter":
				a=6;
				break;
			case " ":
				evalPolicy = !evalPolicy;
				break;
			case "BackSpace":
				switch(task) {
				case "fetchcan":
					task = "docking";
					break;
				case "docking":
					task = "fetchcan";
					break;
				}
				try {
					System.out.println("Load policy parameters for task "+task);
					policy.loadParameters(task);
				} catch(Throwable t) {
					System.out.println("Failed to update policy parameters!");
				}
				break;
			}
		} else {
			a = -1;
		}
		
	}
	
	@Reference
	void setHttpService(HttpService http){
		try {
			// TODO How to register resources with whiteboard pattern?
			http.registerResources("/example/sb", "res", null);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	@Reference
	void setArm(Arm a) {
		this.arm = a;
	}
	
	@Reference
	void setBase(OmniDirectional b) {
		this.base = b;
	}
	
	@Reference
	void setLidar(LaserScanner l) {
		this.lidar = l;
	}
	
	@Reference
	void setSensorStreams(SensorStreams s) {
		this.sensors = s;
	}
	
	@Reference
	void setDiannePlatform(DiannePlatform p) {
		this.diannePlatform = p;
	}
	
	@Reference
	void setDianne(Dianne d) {
		this.dianne = d;
	}
}
