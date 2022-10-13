job "artbot" {
    datacenters = ["dc1"]

    group "artbot" {
        task "bot" {
            driver = "docker"

            config {
                image = "ghcr.io/tag-epic/artbot/artbot:e2d7454d7f4a806fae7a2c3e3aefc3b686256270"
            }

            resources {
                cpu    = 100
                memory = 100
            }
            template {
                data = <<EOF
                    {{ with secret "kv/artbot" }}
                    TOKEN={{.Data.data.token}}
                    {{ end }}
                EOF
                destination = "local/env"
                env         = true
            }
        }
    }
}
