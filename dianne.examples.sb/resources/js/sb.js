
window.addEventListener("keydown", keyevent, false);
window.addEventListener("keyup", keyevent, false);
 
function keyevent(e) {
	 var xhr = new XMLHttpRequest();
	 xhr.open("POST", "/example/sbsse", true);
	 xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
	 xhr.send('type='+e.type+'&key='+e.key+'&code='+e.code);
}


$( document ).ready(function() {
	createLineChart($("#kl"), '', 'Surprise', []);

	var eventsource = new EventSource("/example/sbsse");
	eventsource.onmessage = function(event){
		var result = JSON.parse(event.data);
		
		if(result.prior !== undefined){
			ctx = $('#prior')[0].getContext('2d');
			gauss(result.prior, ctx, 10);
		}
		
		if(result.posterior !== undefined){
			ctx = $('#posterior')[0].getContext('2d');
			gauss(result.posterior, ctx, 10);
		}
		
		if(result.observation !== undefined){
			ctx = $('#observation')[0].getContext('2d');
			render(result.observation, ctx, "laser");
		}
		
		if(result.reconstruction !== undefined){
			ctx = $('#reconstruction')[0].getContext('2d');
			render(result.reconstruction, ctx, "laser");
		}
		
		if(result.kl !== undefined){
			var attr = $("#kl").attr("data-highcharts-chart");
			if(attr!==undefined){
				var chart = Highcharts.charts[Number(attr)];
				if(chart.series.length == 0){
					chart.addSeries({data: []});
				}
				var serie = chart.series[0];
				// shift if the series is longer than 100, higher numbers slow down the browser.
				var shift = serie.data.length > 100;
				// disable animation because adding points can be to fast to complete the animation
				serie.addPoint(result.kl, true, shift, false);
			}
		}

	};	
});

