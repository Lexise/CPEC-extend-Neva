function run() {
  
        // Creating Our XMLHttpRequest object 
        var xhr = new XMLHttpRequest();
  
        // Making our connection  
        var url = 'http://127.0.0.1:8050/';
        xhr.open("GET", url, true);
  
        // function execute after request is successful 
        xhr.onreadystatechange = function () {
            if (this.readyState == 4 && this.status == 200) {
                console.log(this.responseText);
            }
        }
        // Sending our request 
        xhr.send();
    }
    run();