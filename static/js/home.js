var token = "gloryvoicefootprint"

$(document).ready(function() {
    $(".remove").on('click', function() {
        var name = this.id;
        $.ajax({
            url: "/remove",
            type: "POST",
            data: {'token': token, 'name': name},
            success: function(response) {
                alert(JSON.stringify(response));
                window.location.reload("/")
            }
        });
    });

    $(".enroll").on('click', function() {
        var name = $("#name").val();
        var audios = [];

        var frmData = new FormData();
        var audios = $("#audio")[0].files;

        if (name == "") {
            alert("Please input Name");
            return;
        }

        if (audios.length == 0) {
            alert("Please select at least one file");
            return;
        }

        frmData.append('token', token);
        frmData.append('name', name);

        for (var x = 0; x < audios.length; x++) {
            frmData.append("audios", audios[x]);
        }

        $.ajax({
            url: "/enroll",
            type: "POST",
            data: frmData,
            contentType: false,
            processData: false,
            success: function(response) {
                alert(JSON.stringify(response));
                if (response.status == "success") {
                    window.location.reload("/");
                } else {
                }
            }
        });
    });

    $(".verify").on('click', function() {
        var name = $("#name").val();

        var frmData = new FormData();
        var audios = $("#audio")[0].files;

        if (name == "") {
            alert("Please input Name");
            return;
        }

        if (audios.length != 1) {
            alert("Please select one file");
            return;
        }

        frmData.append('token', token);
        frmData.append('name', name);
        frmData.append("audio", audios[0]);

        $.ajax({
            url: "/verify",
            type: "POST",
            data: frmData,
            contentType: false,
            processData: false,
            success: function(response) {
                alert(JSON.stringify(response));
            }
        });
    });

});