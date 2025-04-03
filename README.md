# realtime-metrics

Evaluate recorded GTFS-RT data

## Example usage

[GTFSDB](https://github.com/OpenTransitTools/gtfsdb) (to load the dataset into the db) is needed:


```shell
pip install git+https://github.com/1Maxnet1/gtfsdb.git@patch-1 setuptools # workaround until dependency issues are resolved upstream
```

Now one download a GTFS dataset and load it into the database:

```shell
wget -O BUCHAREST-REGION.zip https://gtfs.tpbi.ro/regional/BUCHAREST-REGION.zip
gtfsdb-load --database_url sqlite:///example.db BUCHAREST-REGION.zip
```

Now install `realtime-metrics` package:

> [!NOTE]  
> Currently GTFSdb works only with SQLAlchemy version 1.x, while realtime-metrics requires SQLAlchemy version 2.x
> Therefore they cannot work in the same environment at the same time

```shell
pip uninstall sqlalchemy
pip install -e .
```

This does also install [gtfsrdb](https://github.com/public-transport/gtfsrdb). 

Afterwards, use gtfsrdb to collect some realtime data:

```shell
gtfsrdb -t https://gtfs.tpbi.ro/api/gtfs-rt/tripUpdates -v https://gtfs.tpbi.ro/api/gtfs-rt/vehiclePositions -d sqlite:///example.db -c
```

Now you can run the script to calculate some metrics on it:

```shell
python ./src/realtime_metrics/main.py -d sqlite:///example.db 
```
